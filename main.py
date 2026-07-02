"""
===========================================================
SMART WEATHER AI — MAIN LOOP
- Polls /station/config every CONFIG_POLL_SECONDS.
- Retrains IMMEDIATELY when the operator changes location
  from the dashboard (new NASA 2-year download for the new
  coordinates).
- Runs a prediction/validation cycle every
  PREDICTION_INTERVAL_SECONDS using live Bresser readings.
===========================================================
"""

import os
import time
import json
from datetime import datetime, timedelta

import config

from api.weather_api import (
    download_weather_data,
    dataset_is_fresh,
    mark_dataset_ready,
)
from preprocessing.preprocess import preprocess_dataset
from training.model_selection import select_best_model

from api.firebase_api import (
    get_live_sensor_data,
    save_prediction,
    append_prediction_history,
    get_station_config,
    set_station_config_defaults,
    save_predictions_all_models,
    get_predictions_all_models,
    save_validation_record,
    append_sensor_history,
    set_pipeline_status,
)

from prediction.predict import (
    predict_next_hour,
    predict_with_all_models,
)


LAST_COORDS_FILE = "models/last_coords.json"

CONFIG_POLL_SECONDS = 60          # how often we check for a location change
STORE_INTERVAL_SECONDS = 1200     # 20 min — store readings + run a cycle
SLOT_MINUTES = 20                 # nested time-slot grid (00, 20, 40)


# ===========================================================
# HELPERS
# ===========================================================

def _slot(dt):
    """Return (date, time) for the nested RTDB store, snapped to the
    20-min grid. e.g. 14:07 -> ('2026-07-02', '14:00'); 14:33 -> '14:20'."""
    minute = (dt.minute // SLOT_MINUTES) * SLOT_MINUTES
    return dt.strftime("%Y-%m-%d"), f"{dt.hour:02d}:{minute:02d}"


def _load_last_coords():
    if not os.path.exists(LAST_COORDS_FILE):
        return None
    try:
        with open(LAST_COORDS_FILE) as f:
            return json.load(f)
    except Exception:
        return None


def _save_last_coords(lat, lon, config_updated_at=""):
    os.makedirs(os.path.dirname(LAST_COORDS_FILE), exist_ok=True)
    with open(LAST_COORDS_FILE, "w") as f:
        json.dump({
            "latitude": lat,
            "longitude": lon,
            "config_updated_at": config_updated_at,
        }, f)


def _coords_changed(new_lat, new_lon, last):
    if last is None:
        return True
    return (
        abs(new_lat - last.get("latitude", 0)) > 1e-4 or
        abs(new_lon - last.get("longitude", 0)) > 1e-4
    )


def _config_retriggered(station, last):
    """True when the operator hit Save & Retrain again — even with the
    exact same coordinates — since the last completed training."""
    if last is None:
        return True
    new_stamp = station.get("updated_at", "")
    return bool(new_stamp) and new_stamp != last.get("config_updated_at", "")


def _sensor_feed_alive(sensor):
    """False when every reading is exactly 0 — Bresser feed dead or paths wrong."""
    return any(abs(float(v)) > 1e-9 for v in sensor.values())


def resolve_station():
    """Read station config from Firebase, seeding a default on first boot."""
    set_station_config_defaults(config.LATITUDE, config.LONGITUDE, "Bengaluru")
    station = get_station_config() or {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "place": "default",
    }
    config.LATITUDE = station["latitude"]
    config.LONGITUDE = station["longitude"]
    return station


def ensure_trained(station):
    """Retrain when coords changed, the operator re-triggered training,
    or models are missing. Skips the NASA download when a fresh (<24h)
    preprocessed dataset for the same coordinates is already on disk."""
    lat, lon = station["latitude"], station["longitude"]
    last = _load_last_coords()
    models_ok = os.path.exists(config.MODEL_PATH)

    needs_training = (
        _coords_changed(lat, lon, last)
        or _config_retriggered(station, last)
        or not models_ok
    )
    if not needs_training:
        return False

    print(f"\n>>> TRAINING for {station['place']} ({lat:.4f}, {lon:.4f})")
    set_pipeline_status("training", station)
    t0 = time.time()
    try:
        if dataset_is_fresh(lat, lon):
            print("Dataset cache hit — skipping NASA download + preprocessing.")
        else:
            download_weather_data(latitude=lat, longitude=lon)
            preprocess_dataset()
            mark_dataset_ready(lat, lon)
        select_best_model()
        _save_last_coords(lat, lon, station.get("updated_at", ""))
        set_pipeline_status("idle", station)
        print(f">>> Training finished in {time.time() - t0:.1f}s")
        return True
    except Exception:
        set_pipeline_status("error", station)
        raise


def run_cycle(station):
    """One prediction + validation cycle against live Bresser data."""
    print("\n" + "=" * 60)
    print(f"PREDICTION CYCLE — {station['place']} "
          f"({station['latitude']:.4f}, {station['longitude']:.4f})")
    print("=" * 60)

    sensor = get_live_sensor_data()
    print("\nLIVE SENSOR DATA")
    for key, value in sensor.items():
        print(f"  {key:15}: {value}")

    if not _sensor_feed_alive(sensor):
        print("\n!! All sensor readings are zero — Bresser feed appears dead.")
        print("   Skipping cycle so validation history stays clean.")
        set_pipeline_status("no_sensor_data", station)
        return

    set_pipeline_status("running", station)

    now = datetime.now()
    date, time = _slot(now)                          # current 20-min slot
    tdate, ttime = _slot(now + timedelta(hours=1))   # slot the forecast targets

    # 1. Store this slot's clean reading — nested sensor_history/{date}/{time}
    append_sensor_history(sensor, date, time)
    print(f"\nStored reading at sensor_history/{date}/{time}")

    # 2. Forecast next hour with all 4 models, keyed to the target slot
    all_preds = predict_with_all_models(
        sensor["temperature"], sensor["humidity"], sensor["wind_speed"],
        sensor["wind_direction"], sensor["rainfall"],
        sensor["pressure"], sensor["irradiance"],
    )
    if all_preds:
        print("NEXT-HOUR PREDICTIONS BY MODEL")
        for name, p in all_preds.items():
            print(f"  {name:20} temp={p['temperature']:.2f} hum={p['humidity']:.2f}")
        save_predictions_all_models(all_preds, tdate, ttime)
    else:
        print("!! No per-model .pkl files found — retrain will fix this.")

    # 3. Close the validation record for THIS slot (prediction made ~1h ago)
    past = get_predictions_all_models(date, time)
    if past and past.get("predictions"):
        save_validation_record(date, time, sensor, past["predictions"])
        print(f"Validation record closed for {date} {time}")
    else:
        print(f"No prior prediction for {date} {time} — first cycle at this slot.")

    # 4. Best-model next-hour forecast + nested prediction history
    prediction = predict_next_hour(
        sensor["temperature"], sensor["humidity"], sensor["wind_speed"],
        sensor["wind_direction"], sensor["rainfall"],
        sensor["pressure"], sensor["irradiance"],
    )
    save_prediction(prediction)
    append_prediction_history(prediction, actual=sensor, date=date, time=time)

    set_pipeline_status("idle", station)
    print("CYCLE COMPLETE")


def run_once():
    """Single-shot entrypoint (Cloud Run Jobs)."""
    station = resolve_station()
    ensure_trained(station)
    run_cycle(station)


# ===========================================================
# MAIN LOOP
# ===========================================================

def main_loop():
    """Continuous loop — used locally and by the Cloud Run service."""
    print("=" * 60)
    print("SMART WEATHER AI — service starting")
    print(f"config poll: {CONFIG_POLL_SECONDS}s · "
          f"store interval: {STORE_INTERVAL_SECONDS // 60}min")
    print("=" * 60)

    last_cycle_at = 0.0

    while True:
        try:
            station = resolve_station()

            # React to a dashboard location change within one poll interval.
            retrained = ensure_trained(station)

            due = (time.time() - last_cycle_at) >= STORE_INTERVAL_SECONDS
            if retrained or due:
                run_cycle(station)
                last_cycle_at = time.time()

        except Exception as e:
            print(f"ERROR: {e}")

        time.sleep(CONFIG_POLL_SECONDS)


if __name__ == "__main__":
    main_loop()

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

# Force Indian Standard Time BEFORE any datetime.now() runs. GitHub Actions
# runs in UTC, which would store a 14:00 IST reading as 08:30 UTC — the wrong
# slot. Setting TZ + tzset makes every naive datetime.now() (slots, timestamps,
# CSV rows, in main.py / firebase_api.py / recorder.py) use the station's local
# clock. IST has no daylight saving, so this is exact year-round.
os.environ["TZ"] = "Asia/Kolkata"
if hasattr(time, "tzset"):
    time.tzset()

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
    append_hourly_trend,
    update_rolling_aggregates,
    set_pipeline_status,
    place_key,
    register_place,
    archive_reading,
    feed_age_seconds,
)

import recorder

from prediction.predict import (
    predict_next_hour,
    predict_with_all_models,
)


LAST_COORDS_FILE = "models/last_coords.json"

CONFIG_POLL_SECONDS = 60          # how often we check for a location change
STALE_FEED_SECONDS = 900          # 15 min — no packet ⇒ station disconnected
SAMPLE_INTERVAL_SECONDS = 300     # 5 min — sample a reading into the excel
STORE_INTERVAL_SECONDS = 1200     # 20 min — aggregate + validate + store + refresh the hourly bucket
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
            "trained_at": datetime.now().isoformat(),
        }, f)


def _model_is_stale(last):
    """True when the model is older than config.RETRAIN_INTERVAL_DAYS —
    triggers the scheduled weekly retrain on the same coordinates."""
    if not last or not last.get("trained_at"):
        return False
    try:
        trained = datetime.fromisoformat(last["trained_at"])
    except (ValueError, TypeError):
        return False
    return (datetime.now() - trained) > timedelta(days=config.RETRAIN_INTERVAL_DAYS)


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


def _feed_is_live(sensor):
    """A reading is only 'live' if it is non-zero AND the webhook received a
    packet recently. Without the freshness check the pipeline would keep
    re-storing the last known value forever after the station disconnects,
    making it look like readings are still arriving."""
    if not _sensor_feed_alive(sensor):
        return False
    age = feed_age_seconds()
    if age is None:
        return False                       # never received a packet
    if age > STALE_FEED_SECONDS:
        print(f"   Station feed is stale ({int(age)}s since last packet) — "
              f"treating as disconnected.")
        return False
    return True


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

    stale = _model_is_stale(last)
    needs_training = (
        _coords_changed(lat, lon, last)
        or _config_retriggered(station, last)
        or not models_ok
        or stale
    )
    if not needs_training:
        return False
    if stale:
        print(f">>> Scheduled weekly retrain (model older than "
              f"{config.RETRAIN_INTERVAL_DAYS} days)")

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


def _window_actual(window_minutes):
    """Aggregated reading over the trailing window from the excel buffer;
    falls back to a single live read if the buffer is empty (fresh start)."""
    actual, n = recorder.aggregate(window_minutes)
    if actual is None:
        live = get_live_sensor_data()
        if not _feed_is_live(live):
            return None, 0
        recorder.record_sample(live)
        return {k: live.get(k, 0) for k in recorder.FIELDS}, 1
    return actual, n


def run_cycle(station):
    """20-min cycle: aggregate the excel window, validate, store (nested)."""
    print("\n" + "=" * 60)
    print(f"20-MIN CYCLE — {station['place']} "
          f"({station['latitude']:.4f}, {station['longitude']:.4f})")
    print("=" * 60)

    actual, n = _window_actual(20)
    if actual is None:
        print("!! Station not connected (no fresh packet). Skipping — "
              "nothing stored to Firebase this cycle.")
        set_pipeline_status("no_sensor_data", station)
        return

    print(f"Aggregated actual over {n} sample(s): "
          f"temp={actual['temperature']:.2f} hum={actual['humidity']:.2f}")
    set_pipeline_status("running", station)

    now = datetime.now()
    date, time = _slot(now)                          # current 20-min slot
    # Forecast the NEXT 20-min slot so validation closes every 20 min
    # (matches the mentor's 20-min actual-vs-predicted cadence) instead of
    # only once an hour.
    tdate, ttime = _slot(now + timedelta(minutes=SLOT_MINUTES))

    # Per-place archive: register the place and tag this reading to it so the
    # dashboard can export CSVs by place + date.
    pk = place_key(station["latitude"], station["longitude"])
    register_place(pk, station.get("place", ""), station["latitude"], station["longitude"])

    # 1. Store the aggregated reading — nested sensor_history/{date}/{time}
    append_sensor_history(actual, date, time, samples=n)
    archive_reading(pk, "readings", date, time,
                    {**actual, "samples": n, "timestamp": now.isoformat()})
    print(f"Stored reading at sensor_history/{date}/{time} (place {pk})")

    # 2. Forecast next hour with all 4 models, keyed to the target slot
    all_preds = predict_with_all_models(
        actual["temperature"], actual["humidity"], actual["wind_speed"],
        actual["wind_direction"], actual["rainfall"],
        actual["pressure"], actual["irradiance"],
    )
    if all_preds:
        save_predictions_all_models(all_preds, tdate, ttime)
    else:
        print("!! No per-model .pkl files found — retrain will fix this.")

    # 3. Close the validation record for THIS slot (prediction made ~1h ago)
    past = get_predictions_all_models(date, time)
    if past and past.get("predictions"):
        save_validation_record(date, time, actual, past["predictions"])
        archive_reading(pk, "validation", date, time,
                        {"actual": actual, "predictions": past["predictions"],
                         "timestamp": now.isoformat()})
        print(f"Validation record closed for {date} {time}")
    else:
        print(f"No prior prediction for {date} {time} — first cycle at this slot.")

    # 4. Best-model forecast + nested prediction history + results excel
    prediction = predict_next_hour(
        actual["temperature"], actual["humidity"], actual["wind_speed"],
        actual["wind_direction"], actual["rainfall"],
        actual["pressure"], actual["irradiance"],
    )
    save_prediction(prediction)
    append_prediction_history(prediction, actual=actual, date=date, time=time)
    recorder.write_result("twentymin", actual, prediction)

    set_pipeline_status("idle", station)
    print("20-MIN CYCLE COMPLETE")


def run_hourly(station):
    """Refresh the CURRENT real clock hour's trailing-60-min aggregate.
    Called every 20-min cycle (not on its own timer) so the bucket for
    hour "15:00" always exists within 20 min of the station being live at
    that hour — see main_loop() for why a separate timer was unreliable."""
    actual, n = recorder.aggregate(60)
    if actual is None:
        return
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    hour = f"{now.hour:02d}:00"

    prediction = None
    try:
        prediction = predict_next_hour(
            actual["temperature"], actual["humidity"], actual["wind_speed"],
            actual["wind_direction"], actual["rainfall"],
            actual["pressure"], actual["irradiance"],
        )
    except Exception:
        pass

    append_hourly_trend(actual, date, hour, samples=n, predicted=prediction)

    # Per-place hourly archive for CSV export.
    pk = place_key(station["latitude"], station["longitude"])
    register_place(pk, station.get("place", ""), station["latitude"], station["longitude"])
    archive_reading(pk, "hourly", date, hour,
                    {**actual, "samples": n, "predicted": prediction,
                     "timestamp": now.isoformat()})

    recorder.write_result("hourly", actual, prediction)
    print(f"Hourly trend stored at hourly_trend/{date}/{hour} ({n} samples)")

    # Refresh daily/weekly/monthly summary cards from live history.
    try:
        update_rolling_aggregates()
        print("Rolling aggregates (daily/weekly/monthly) refreshed.")
    except Exception as e:
        print(f"Warning: could not update rolling aggregates: {e}")


def _sample_now():
    """Take one live reading into the excel buffer (only if the station is
    actually connected — a fresh, non-zero packet)."""
    live = get_live_sensor_data()
    if _feed_is_live(live):
        recorder.record_sample(live)
        return True
    return False


def run_once():
    """Single-shot entrypoint (Cloud Run Jobs): sample, 20-min cycle, hourly."""
    station = resolve_station()
    ensure_trained(station)
    _sample_now()
    run_cycle(station)
    run_hourly(station)


# ===========================================================
# MAIN LOOP
# ===========================================================

def main_loop(max_runtime_seconds=None):
    """Continuous loop — used locally, by the Cloud Run service, and by the
    GitHub Actions runner.

    Two cadences off one 60s poll:
      • sample the excel every 10 min
      • aggregate + validate + store every 20 min — run_hourly() is called
        in the SAME tick, refreshing the current real clock hour's bucket
        (e.g. "15:00") with the trailing 60-min aggregate. Firing it off a
        real clock-hour-aligned event (not a standalone 3600s timer) matters
        because this loop restarts often (every ~6h automatically, and on
        every redeploy) — a timer seeded at 0 fires once immediately on
        startup then drifts +3600s from THAT moment, never landing on real
        :00 boundaries. Piggy-backing on the 20-min cycle is restart-safe:
        each restart still lands on a real 20-min slot, and hourly_trend for
        the current hour is refreshed up to 3 times before the hour rolls
        over, converging to a near-complete hourly average.

    When max_runtime_seconds is set (GitHub Actions caps a single job at 6h),
    the loop exits cleanly with status 0 after that budget so a scheduled
    restart can take over. All durable state (predictions awaiting validation,
    nested history) already lives in RTDB, so the handoff is seamless — the
    next run retrains only if the coordinates changed and continues the cycle.
    """
    print("=" * 60)
    print("SMART WEATHER AI — service starting")
    print(f"poll {CONFIG_POLL_SECONDS}s · sample {SAMPLE_INTERVAL_SECONDS // 60}min "
          f"· cycle+hourly {STORE_INTERVAL_SECONDS // 60}min")
    print("=" * 60)

    started_at = time.time()
    last_sample_at = 0.0
    last_cycle_at = 0.0

    while True:
        try:
            station = resolve_station()

            # React to a dashboard location change within one poll interval.
            retrained = ensure_trained(station)

            now = time.time()

            if retrained or (now - last_sample_at) >= SAMPLE_INTERVAL_SECONDS:
                if _sample_now():
                    print("· sampled a reading into the excel")
                last_sample_at = now

            if retrained or (now - last_cycle_at) >= STORE_INTERVAL_SECONDS:
                run_cycle(station)
                run_hourly(station)
                last_cycle_at = now

        except Exception as e:
            print(f"ERROR: {e}")

        if max_runtime_seconds and (time.time() - started_at) >= max_runtime_seconds:
            print(f">>> Reached max runtime ({max_runtime_seconds}s) — exiting "
                  f"cleanly for the scheduled restart.")
            return

        time.sleep(CONFIG_POLL_SECONDS)


if __name__ == "__main__":
    # On GitHub Actions we pass MAX_RUNTIME_SECONDS so the loop exits before the
    # 6h job cap and the schedule restarts it. Locally it stays unset → runs
    # forever, exactly as before.
    _budget = int(os.environ.get("MAX_RUNTIME_SECONDS", "0")) or None
    main_loop(max_runtime_seconds=_budget)

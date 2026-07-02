"""
===========================================================
SMART WEATHER AI — MAIN LOOP
Reads station config (lat/lon/place) from Firebase.
Retrains NASA pipeline when coordinates change.
Every cycle: writes clean live reading, forecasts with all
4 models, and closes previous hour's validation record.
===========================================================
"""

import os
import time
import json
from datetime import datetime, timedelta

import config

from api.weather_api import download_weather_data
from preprocessing.preprocess import preprocess_dataset
from training.model_selection import select_best_model

from api.firebase_api import (
    get_live_sensor_data,
    save_prediction,
    save_validation,
    append_prediction_history,
    get_station_config,
    set_station_config_defaults,
    append_live_reading,
    save_predictions_all_models,
    get_predictions_all_models,
    save_validation_record,
    trim_validation_history,
)

from prediction.predict import (
    predict_next_hour,
    predict_with_all_models,
)


LAST_COORDS_FILE = "models/last_coords.json"


def _hour_key(dt):
    return dt.strftime("%Y-%m-%dT%H")


def _load_last_coords():
    if not os.path.exists(LAST_COORDS_FILE):
        return None
    try:
        with open(LAST_COORDS_FILE) as f:
            return json.load(f)
    except Exception:
        return None


def _save_last_coords(lat, lon):
    os.makedirs(os.path.dirname(LAST_COORDS_FILE), exist_ok=True)
    with open(LAST_COORDS_FILE, "w") as f:
        json.dump({"latitude": lat, "longitude": lon}, f)


def _coords_changed(new_lat, new_lon, last):
    if last is None:
        return True
    return (
        abs(new_lat - last.get("latitude", 0)) > 1e-4 or
        abs(new_lon - last.get("longitude", 0)) > 1e-4
    )


def _run_training(lat, lon):
    print(f"\n>>> Training pipeline for {lat:.4f}, {lon:.4f}")
    download_weather_data(latitude=lat, longitude=lon)
    preprocess_dataset()
    select_best_model()
    _save_last_coords(lat, lon)


def main():
    print("=" * 60)
    print("SMART WEATHER AI")
    print("=" * 60)

    # -----------------------------------------------------------
    # 1. RESOLVE STATION LOCATION
    # -----------------------------------------------------------
    set_station_config_defaults(config.LATITUDE, config.LONGITUDE, "Bengaluru")

    station = get_station_config() or {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "place": "default",
    }
    lat = station["latitude"]
    lon = station["longitude"]
    place = station["place"]
    print(f"\nStation: {place} ({lat:.4f}, {lon:.4f})")

    # -----------------------------------------------------------
    # Sync in-memory config so downstream modules pick up the value
    # -----------------------------------------------------------
    config.LATITUDE = lat
    config.LONGITUDE = lon

    # -----------------------------------------------------------
    # 2. RE-TRAIN IF COORDS CHANGED OR MODELS MISSING
    # -----------------------------------------------------------
    last = _load_last_coords()
    models_ok = os.path.exists(config.MODEL_PATH)
    if _coords_changed(lat, lon, last) or not models_ok:
        _run_training(lat, lon)
    else:
        print("Models already trained for these coordinates. Skipping training.")

    # -----------------------------------------------------------
    # 3. READ LIVE SENSOR DATA
    # -----------------------------------------------------------
    sensor = get_live_sensor_data()
    print("\nLIVE SENSOR DATA")
    for key, value in sensor.items():
        print(f"  {key:15}: {value}")

    now = datetime.now()
    this_hour_key = _hour_key(now)
    next_hour_key = _hour_key(now + timedelta(hours=1))

    # -----------------------------------------------------------
    # 4. STORE CLEAN LIVE READING FOR THIS HOUR
    # -----------------------------------------------------------
    append_live_reading(sensor, this_hour_key)

    # -----------------------------------------------------------
    # 5. PREDICT NEXT HOUR — ALL 4 MODELS
    # -----------------------------------------------------------
    all_preds = predict_with_all_models(
        sensor["temperature"], sensor["humidity"], sensor["wind_speed"],
        sensor["wind_direction"], sensor["rainfall"],
        sensor["pressure"], sensor["irradiance"],
    )
    print("\nNEXT-HOUR PREDICTIONS BY MODEL")
    for name, p in all_preds.items():
        print(f"  {name:20} temp={p['temperature']:.2f} hum={p['humidity']:.2f}")

    save_predictions_all_models(all_preds, next_hour_key)

    # -----------------------------------------------------------
    # 6. CLOSE THIS HOUR'S VALIDATION RECORD
    #    (this hour's actual  ↔  prediction written last hour)
    # -----------------------------------------------------------
    past_predictions = get_predictions_all_models(this_hour_key)
    if past_predictions and past_predictions.get("predictions"):
        save_validation_record(this_hour_key, sensor, past_predictions["predictions"])
        print(f"\nValidation record closed for hour {this_hour_key}")
    else:
        print(f"\nNo prior prediction for {this_hour_key} — first cycle.")

    trim_validation_history(200)

    # -----------------------------------------------------------
    # 7. LEGACY WRITES (best-model prediction + history)
    # -----------------------------------------------------------
    prediction = predict_next_hour(
        sensor["temperature"], sensor["humidity"], sensor["wind_speed"],
        sensor["wind_direction"], sensor["rainfall"],
        sensor["pressure"], sensor["irradiance"],
    )
    save_prediction(prediction)
    save_validation(sensor, prediction)
    append_prediction_history(prediction, actual=sensor)

    print("\nCYCLE COMPLETE")


# ===========================================================
if __name__ == "__main__":
    SLEEP_SECONDS = 1800  # 30 min

    while True:
        try:
            main()
        except Exception as e:
            print(f"ERROR: {e}")

        print(f"\nSleeping {SLEEP_SECONDS // 60} min…\n")
        time.sleep(SLEEP_SECONDS)

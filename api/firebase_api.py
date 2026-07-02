import os
import firebase_admin
from datetime import datetime
from firebase_admin import credentials
from firebase_admin import db

import config

# ==========================================================
# INITIALIZE FIREBASE
#   - Local dev: use serviceAccountKey.json if present.
#   - Cloud Run: no key file — fall back to Application Default
#     Credentials, i.e. the service the container runs as. This
#     keeps secrets OUT of the image.
# ==========================================================

if not firebase_admin._apps:

    options = {"databaseURL": config.FIREBASE_DATABASE_URL}

    if os.path.exists(config.SERVICE_ACCOUNT_FILE):
        cred = credentials.Certificate(config.SERVICE_ACCOUNT_FILE)
    else:
        # ApplicationDefault() picks up the Cloud Run runtime service
        # account automatically (no key material on disk).
        cred = credentials.ApplicationDefault()

    firebase_admin.initialize_app(cred, options)

# ==========================================================
# READ VALUE
# ==========================================================

def read_value(path, default=0):

    try:

        value = db.reference(path).get()

        if value is None:

            return default

        return float(value)

    except:

        return default

# ==========================================================
# READ LIVE SENSOR DATA
# ==========================================================

def get_live_sensor_data():
    """Read the live Bresser reading from weather_station/payload — the same
    node the dashboard uses — falling back to the legacy analytics/* paths
    for any field the payload doesn't carry."""

    payload = {}
    try:
        node = db.reference("weather_station").get() or {}
        payload = node.get("payload", node) or {}
    except Exception:
        payload = {}

    def pick(*keys, fallback_path=None):
        for k in keys:
            v = payload.get(k)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
        return read_value(fallback_path) if fallback_path else 0.0

    return {
        "temperature":    pick("temperature", fallback_path="analytics/daily/avg_temperature"),
        "humidity":       pick("humidity", fallback_path="analytics/daily/avg_humidity"),
        "wind_speed":     pick("wind_speed", "wind_avg_ms", fallback_path="analytics/wind/average_speed"),
        "wind_direction": pick("wind_direction", "wind_dir", fallback_path="analytics/wind/average_direction"),
        "rainfall":       pick("rain", "rainfall", fallback_path="analytics/rainfall/rainfall_percentage"),
        "pressure":       pick("pressure", fallback_path="analytics/daily/avg_pressure"),
        "irradiance":     pick("irradiance", "light", fallback_path="analytics/light/avg_light"),
    }

# ==========================================================
# WRITE PREDICTIONS
# ==========================================================

def save_prediction(prediction):

    db.reference(

        "prediction/next_hour"

    ).set(

        prediction

    )


# ==========================================================
# APPEND PREDICTION TO ROLLING HISTORY (for dashboard time-series)
# ==========================================================

def append_prediction_history(prediction, actual=None, max_entries=200):

    entry = {
        "timestamp": datetime.now().isoformat(),
        "predicted": prediction,
    }

    if actual is not None:
        entry["actual"] = actual

    ref = db.reference("prediction_history")
    ref.push(entry)

    all_entries = ref.get() or {}

    if len(all_entries) > max_entries:

        oldest_keys = sorted(all_entries.keys())[: len(all_entries) - max_entries]

        for key in oldest_keys:
            ref.child(key).delete()


# ==========================================================
# SAVE ALL-MODEL METRICS (for dashboard comparison charts)
# ==========================================================

# ==========================================================
# STATION CONFIG (lat/lon/place) — read from dashboard input
# ==========================================================

def get_station_config():
    """Return {'latitude', 'longitude', 'place'} from /station/config, or None."""
    cfg = db.reference("station/config").get()
    if not cfg:
        return None
    return {
        "latitude": float(cfg.get("latitude", 0)),
        "longitude": float(cfg.get("longitude", 0)),
        "place": cfg.get("place", ""),
    }


def set_station_config_defaults(latitude, longitude, place="Default"):
    """Write initial config only if none exists."""
    ref = db.reference("station/config")
    if not ref.get():
        ref.set({"latitude": latitude, "longitude": longitude, "place": place})


# ==========================================================
# PIPELINE STATUS — lets the dashboard show training progress
# ==========================================================

def set_pipeline_status(state, station=None):
    """state: idle | training | running | no_sensor_data | error"""
    payload = {
        "state": state,
        "updated_at": datetime.now().isoformat(),
    }
    if station:
        payload["place"] = station.get("place", "")
        payload["latitude"] = station.get("latitude", 0)
        payload["longitude"] = station.get("longitude", 0)
    try:
        db.reference("station/status").set(payload)
    except Exception as e:
        print(f"Warning: could not write pipeline status: {e}")


# ==========================================================
# HOURLY LIVE READING BUCKET (Bresser sensor data, cleaned)
# ==========================================================

def append_live_reading(sensor, hour_key):
    """Overwrite live_readings/{hour_key} with a clean reading (no rssi/sensor_id)."""
    clean = {k: sensor.get(k, 0) for k in [
        "temperature", "humidity", "wind_speed", "wind_direction",
        "rainfall", "pressure", "irradiance"
    ]}
    clean["timestamp"] = datetime.now().isoformat()
    db.reference(f"live_readings/{hour_key}").set(clean)


# ==========================================================
# ALL-MODEL PREDICTIONS BUCKET (forecast for a target hour)
# ==========================================================

def save_predictions_all_models(predictions_by_model, target_hour_key):
    db.reference(f"predictions_all_models/{target_hour_key}").set({
        "predicted_for": target_hour_key,
        "written_at": datetime.now().isoformat(),
        "predictions": predictions_by_model,
    })


def get_predictions_all_models(hour_key):
    return db.reference(f"predictions_all_models/{hour_key}").get()


# ==========================================================
# VALIDATION HISTORY — actual vs 4-model predictions per hour
# ==========================================================

def save_validation_record(hour_key, actual, predictions_by_model):
    clean_actual = {k: actual.get(k, 0) for k in [
        "temperature", "humidity", "wind_speed", "wind_direction",
        "rainfall", "pressure", "irradiance"
    ]}
    db.reference(f"validation_history/{hour_key}").set({
        "hour": hour_key,
        "actual": clean_actual,
        "predictions": predictions_by_model,
    })


def trim_validation_history(max_entries=200):
    ref = db.reference("validation_history")
    entries = ref.get() or {}
    if len(entries) > max_entries:
        oldest = sorted(entries.keys())[: len(entries) - max_entries]
        for k in oldest:
            ref.child(k).delete()


def save_model_metrics(results, best_name):

    payload = {
        "_best": best_name,
        "_updated_at": datetime.now().isoformat(),
    }

    for row in results:
        payload[row["Model"]] = {
            "train_r2": row["Train R2"],
            "test_r2":  row["Test R2"],
            "mae":      row["MAE"],
            "mse":      row["MSE"],
            "rmse":     row["RMSE"],
            "samples":  row.get("samples", []),
        }

    db.reference("model_metrics").set(payload)
# ==========================================================
# SAVE VALIDATION DATA
# ==========================================================

def save_validation(actual, predicted):

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    db.reference(

        f"validation/{timestamp}"

    ).set(

        {

            "actual": {

                "temperature": actual["temperature"],

                "humidity": actual["humidity"],

                "wind_speed": actual["wind_speed"],

                "wind_direction": actual["wind_direction"],

                "rainfall": actual["rainfall"],

                "pressure": actual["pressure"],

                "irradiance": actual["irradiance"]

            },
            "predicted": {

                "temperature": predicted["temperature"],

                "humidity": predicted["humidity"],

                "wind_speed": predicted["wind_speed"],

                "wind_direction": predicted["wind_direction"],

                "rainfall": predicted["rainfall"],

                "pressure": predicted["pressure"],

                "irradiance": predicted["irradiance"]

            }
            

        }

    )

    print("Validation data saved.")
# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    sensor = get_live_sensor_data()

    print(sensor)

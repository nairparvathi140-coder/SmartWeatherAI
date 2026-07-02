import os
import firebase_admin
from datetime import datetime, timedelta
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
# NESTED PREDICTION HISTORY (date → time, for dashboard time-series)
# ==========================================================

def append_prediction_history(prediction, actual=None, date=None, time=None):

    entry = {
        "timestamp": datetime.now().isoformat(),
        "predicted": prediction,
    }
    if actual is not None:
        entry["actual"] = actual

    # Fall back to the current slot if the caller didn't pass one.
    if date is None or time is None:
        now = datetime.now()
        minute = (now.minute // 20) * 20
        date = now.strftime("%Y-%m-%d")
        time = f"{now.hour:02d}:{minute:02d}"

    db.reference(f"prediction_history/{date}/{time}").set(entry)
    _trim_nested_dates("prediction_history")


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


SENSOR_FIELDS = [
    "temperature", "humidity", "wind_speed", "wind_direction",
    "rainfall", "pressure", "irradiance",
]


def _clean_reading(sensor):
    return {k: round(float(sensor.get(k, 0)), 4) for k in SENSOR_FIELDS}


def _trim_nested_dates(path, keep_days=14):
    """Nested stores are {path}/{date}/{time}. Keep only the newest
    `keep_days` date nodes so the tree can't grow without bound."""
    ref = db.reference(path)
    dates = ref.get(shallow=True) or {}
    if len(dates) > keep_days:
        for d in sorted(dates.keys())[: len(dates) - keep_days]:
            ref.child(d).delete()


# ==========================================================
# NESTED SENSOR HISTORY  (date → time → reading, every 20 min)
# ==========================================================

def append_sensor_history(sensor, date, time, samples=None):
    """Store a clean reading at sensor_history/{date}/{time}.
    The date→time nesting is what renders as a scroll-down tree in the
    Firebase console. Overwriting the same (date,time) slot is idempotent."""
    clean = _clean_reading(sensor)
    clean["timestamp"] = datetime.now().isoformat()
    if samples is not None:
        clean["samples"] = samples
    db.reference(f"sensor_history/{date}/{time}").set(clean)
    _trim_nested_dates("sensor_history")


# ==========================================================
# NESTED HOURLY TREND  (date → HH:00 → hourly-averaged reading)
# ==========================================================

def update_rolling_aggregates():
    """Recompute analytics/{daily,weekly,monthly} from sensor_history so the
    dashboard summary cards reflect live data instead of the legacy system's
    stale values. Rolling windows: daily=24h, weekly=7d, monthly=30d."""
    tree = db.reference("sensor_history").get() or {}
    rows = []
    for date, times in tree.items():
        if not isinstance(times, dict):
            continue
        for t, reading in times.items():
            try:
                dt = datetime.strptime(f"{date} {t}", "%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                continue
            if isinstance(reading, dict):
                rows.append((dt, reading))

    if not rows:
        return

    now = datetime.now()

    def _agg(days, label):
        cutoff = now - timedelta(days=days)
        window = [r for (dt, r) in rows if dt >= cutoff]
        if not window:
            return
        out = {"total_readings": len(window), "last_updated": now.isoformat()}
        for p in ["temperature", "humidity", "pressure"]:
            vals = [float(r[p]) for r in window
                    if isinstance(r.get(p), (int, float))]
            if vals:
                out[f"avg_{p}"] = round(sum(vals) / len(vals), 2)
                out[f"max_{p}"] = round(max(vals), 2)
                out[f"min_{p}"] = round(min(vals), 2)
        if label == "daily":
            rain = [float(r["rainfall"]) for r in window
                    if isinstance(r.get("rainfall"), (int, float))]
            rain_events = sum(1 for v in rain if v > 0)
            db.reference("analytics/rainfall").update({
                "rain_events": rain_events,
                "no_rain_events": len(rain) - rain_events,
                "rainfall_percentage": round(100 * rain_events / len(rain), 2) if rain else 0,
            })
        db.reference(f"analytics/{label}").update(out)

    _agg(1, "daily")
    _agg(7, "weekly")
    _agg(30, "monthly")


def append_hourly_trend(sensor, date, hour, samples=None, predicted=None):
    """Store an hour's aggregated reading at hourly_trend/{date}/{hour}."""
    clean = _clean_reading(sensor)
    clean["timestamp"] = datetime.now().isoformat()
    if samples is not None:
        clean["samples"] = samples
    if predicted is not None:
        clean["predicted"] = predicted
    db.reference(f"hourly_trend/{date}/{hour}").set(clean)
    _trim_nested_dates("hourly_trend", keep_days=30)


# ==========================================================
# ALL-MODEL PREDICTIONS BUCKET (forecast for a target date/time slot)
# ==========================================================

def save_predictions_all_models(predictions_by_model, target_date, target_time):
    db.reference(f"predictions_all_models/{target_date}/{target_time}").set({
        "predicted_for": f"{target_date} {target_time}",
        "written_at": datetime.now().isoformat(),
        "predictions": predictions_by_model,
    })


def get_predictions_all_models(date, time):
    return db.reference(f"predictions_all_models/{date}/{time}").get()


# ==========================================================
# NESTED VALIDATION HISTORY — actual vs 4-model predictions per slot
# ==========================================================

def save_validation_record(date, time, actual, predictions_by_model):
    db.reference(f"validation_history/{date}/{time}").set({
        "hour": f"{date} {time}",
        "actual": _clean_reading(actual),
        "predictions": predictions_by_model,
    })
    _trim_nested_dates("validation_history")
    _trim_nested_dates("predictions_all_models")


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

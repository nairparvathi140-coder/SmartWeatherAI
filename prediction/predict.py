"""
===========================================================
PREDICTION ENGINE
===========================================================
"""

import os
import joblib
import numpy as np
import pandas as pd

import config


# ===========================================================
# INPUT SANITIZATION (ML defense-in-depth)
# ===========================================================
# Even though the EMQX webhook validates physical bounds at ingest, the model
# clamps its OWN inputs too. This stops a poisoned/faulty reading that somehow
# reaches the pipeline (bypassed webhook, corrupt analytics node, sensor
# fault) from driving the model to absurd extrapolations. NaN/inf → midpoint.

_INPUT_BOUNDS = {
    "temperature": (-60.0, 65.0, 20.0),
    "humidity": (0.0, 100.0, 50.0),
    "wind_speed": (0.0, 120.0, 0.0),
    "rainfall": (0.0, 500.0, 0.0),
    "irradiance": (0.0, 1500.0, 0.0),
    "pressure_hpa": (800.0, 1100.0, 1013.0),
}


def _clamp(name, value):
    lo, hi, default = _INPUT_BOUNDS[name]
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(v) or np.isinf(v):
        return default
    return min(hi, max(lo, v))


# ===========================================================
# FEATURE VECTOR BUILDER (shared)
# ===========================================================

def _build_features(
    temperature, humidity, wind_speed,
    wind_direction_degrees, rainfall, pressure, irradiance,
):
    # Wind direction: wrap to [0,360) then encode; NaN → 0°.
    try:
        wd = float(wind_direction_degrees)
        if np.isnan(wd) or np.isinf(wd):
            wd = 0.0
    except (TypeError, ValueError):
        wd = 0.0
    radians = np.deg2rad(wd % 360.0)
    wind_dir_sin = float(np.sin(radians))
    wind_dir_cos = float(np.cos(radians))

    # Pressure unit auto-detect: NASA training data is hPa (~850-1100).
    # Bresser/analytics feeds may report kPa (~85-110) or hPa directly.
    try:
        praw = float(pressure)
    except (TypeError, ValueError):
        praw = 1013.0
    pressure_hpa = praw * 10 if praw < 200 else praw

    # Clamp every model input to its physical range.
    temperature = _clamp("temperature", temperature)
    humidity = _clamp("humidity", humidity)
    wind_speed = _clamp("wind_speed", wind_speed)
    rainfall = _clamp("rainfall", rainfall)
    irradiance = _clamp("irradiance", irradiance)
    pressure_hpa = _clamp("pressure_hpa", pressure_hpa)

    # Named DataFrame so columns align with training exactly
    return pd.DataFrame(
        [[
            temperature, humidity, wind_speed,
            wind_dir_sin, wind_dir_cos,
            rainfall, pressure_hpa, irradiance,
        ]],
        columns=config.FEATURE_COLUMNS,
    )


# ===========================================================
# LOAD MODEL
# ===========================================================

def load_model():

    if not os.path.exists(config.MODEL_PATH):

        raise FileNotFoundError(
            "Model not found.\nRun model_selection.py first."
        )

    return joblib.load(config.MODEL_PATH)


# ===========================================================
# PREDICT NEXT HOUR
# ===========================================================

def predict_next_hour(

    temperature,

    humidity,

    wind_speed,

    wind_direction_degrees,

    rainfall,

    pressure,

    irradiance

):

    model = load_model()

    features = _build_features(
        temperature, humidity, wind_speed,
        wind_direction_degrees, rainfall, pressure, irradiance,
    )

    prediction = model.predict(features)[0]

    # -------------------------------------------------------
    # Post-process predictions
    # -------------------------------------------------------

    temperature = float(prediction[0])

    humidity = float(prediction[1])

    wind_speed = max(0, float(prediction[2]))

    wind_dir_sin = float(np.clip(prediction[3], -1, 1))

    wind_dir_cos = float(np.clip(prediction[4], -1, 1))

    # -------------------------------------------------------
    # Convert Sin/Cos back to Degrees
    # -------------------------------------------------------

    wind_direction = np.degrees(

        np.arctan2(

            wind_dir_sin,

            wind_dir_cos

        )

    )

    if wind_direction < 0:

        wind_direction += 360

    rainfall = max(0, float(prediction[5]))

    pressure = float(np.clip(prediction[6], 850, 1100))

    irradiance = max(0, float(prediction[7]))

    return {

        "temperature": temperature,

        "humidity": humidity,

        "wind_speed": wind_speed,

        "wind_direction": float(wind_direction),

        "rainfall": rainfall,

        "pressure": pressure,

        "irradiance": irradiance

    }
# ===========================================================
# PREDICT WITH ALL 4 MODELS
# ===========================================================

def _postprocess(prediction):
    temperature = float(prediction[0])
    humidity = float(prediction[1])
    wind_speed = max(0, float(prediction[2]))
    sin_v = float(np.clip(prediction[3], -1, 1))
    cos_v = float(np.clip(prediction[4], -1, 1))
    wind_direction = float(np.degrees(np.arctan2(sin_v, cos_v)))
    if wind_direction < 0:
        wind_direction += 360
    rainfall = max(0, float(prediction[5]))
    pressure = float(np.clip(prediction[6], 850, 1100))
    irradiance = max(0, float(prediction[7]))
    return {
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "wind_direction": wind_direction,
        "rainfall": rainfall,
        "pressure": pressure,
        "irradiance": irradiance,
    }


def predict_with_all_models(
    temperature,
    humidity,
    wind_speed,
    wind_direction_degrees,
    rainfall,
    pressure,
    irradiance,
):
    """Load every {name}.pkl from models/ and return {model_name: prediction_dict}."""
    features = _build_features(
        temperature, humidity, wind_speed,
        wind_direction_degrees, rainfall, pressure, irradiance,
    )

    predictions = {}
    for name in config.MODELS:
        slug = name.lower().replace(" ", "_") + ".pkl"
        path = os.path.join(config.MODEL_FOLDER, slug)
        if not os.path.exists(path):
            continue
        model = joblib.load(path)
        raw = model.predict(features)[0]
        predictions[name] = _postprocess(raw)

    return predictions


# ===========================================================

if __name__ == "__main__":

    result = predict_next_hour(

        temperature=28,

        humidity=65,

        wind_speed=3,

        wind_direction_degrees=180,

        rainfall=0,

        pressure=101,

        irradiance=650

    )

    print("\nPrediction\n")

    for key, value in result.items():

        print(f"{key:15} : {value:.3f}")

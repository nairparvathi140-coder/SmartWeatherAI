"""
===========================================================
PREDICTION ENGINE
===========================================================
"""

import os
import joblib
import numpy as np

import config


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

    # ----------------------------------------------
    # Wind Direction
    # Degrees → Sin/Cos
    # ----------------------------------------------

    radians = np.deg2rad(wind_direction_degrees)

    wind_dir_sin = np.sin(radians)

    wind_dir_cos = np.cos(radians)

    # ----------------------------------------------
    # Pressure
    # kPa → hPa
    # ----------------------------------------------

    pressure_hpa = pressure * 10

    # ----------------------------------------------
    # Create Feature Vector
    # ----------------------------------------------

    features = [[

        temperature,

        humidity,

        wind_speed,

        wind_dir_sin,

        wind_dir_cos,

        rainfall,

        pressure_hpa,

        irradiance

    ]]

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

def _postprocess(prediction, wind_dir_sin, wind_dir_cos):
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
    radians = np.deg2rad(wind_direction_degrees)
    wind_dir_sin = float(np.sin(radians))
    wind_dir_cos = float(np.cos(radians))
    pressure_hpa = pressure * 10

    features = [[
        temperature, humidity, wind_speed,
        wind_dir_sin, wind_dir_cos,
        rainfall, pressure_hpa, irradiance,
    ]]

    predictions = {}
    for name in config.MODELS:
        slug = name.lower().replace(" ", "_") + ".pkl"
        path = os.path.join(config.MODEL_FOLDER, slug)
        if not os.path.exists(path):
            continue
        model = joblib.load(path)
        raw = model.predict(features)[0]
        predictions[name] = _postprocess(raw, wind_dir_sin, wind_dir_cos)

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

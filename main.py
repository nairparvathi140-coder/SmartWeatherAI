"""
===========================================================
SMART WEATHER AI
MAIN PROGRAM
===========================================================
"""

import config

from api.weather_api import download_weather_data
from preprocessing.preprocess import preprocess_dataset
from training.model_selection import select_best_model

from api.firebase_api import (
    get_live_sensor_data,
    save_prediction,
    save_validation
)

from prediction.predict import predict_next_hour


# ===========================================================
# MAIN
# ===========================================================

def main():

    print("=" * 60)
    print("SMART WEATHER AI")
    print("=" * 60)

    # -------------------------------------------------------
    # STEP 1
    # Download NASA Dataset
    # -------------------------------------------------------

    download_weather_data()

    # -------------------------------------------------------
    # STEP 2
    # Preprocess Dataset
    # -------------------------------------------------------

    preprocess_dataset()

    # -------------------------------------------------------
    # STEP 3
    # Train / Load Best Model
    # -------------------------------------------------------

    select_best_model()

    # -------------------------------------------------------
    # STEP 4
    # Read Live ESP32 Data
    # -------------------------------------------------------

    sensor = get_live_sensor_data()

    print("\nLIVE SENSOR DATA\n")

    for key, value in sensor.items():

        print(f"{key:15} : {value}")

    # -------------------------------------------------------
    # STEP 5
    # Prediction
    # -------------------------------------------------------

    prediction = predict_next_hour(

        sensor["temperature"],

        sensor["humidity"],

        sensor["wind_speed"],

        sensor["wind_direction"],

        sensor["rainfall"],

        sensor["pressure"],

        sensor["irradiance"]

    )

    print("\nPREDICTED NEXT HOUR\n")

    for key, value in prediction.items():

        print(f"{key:15} : {value:.3f}")

    # -------------------------------------------------------
    # STEP 6
    # Save Prediction
    # -------------------------------------------------------

    save_prediction(

        prediction

    )

    # -------------------------------------------------------
    # STEP 7
    # Save Validation Data
    # -------------------------------------------------------

    save_validation(

        sensor,

        prediction

    )

    print("\nPrediction saved successfully.")

    print("Validation data saved successfully.")

    print("\nSYSTEM COMPLETED")


# ===========================================================
import time

if __name__ == "__main__":

    while True:

        try:

            main()

        except Exception as e:

            print(e)

        print("\nWaiting 30 minutes...\n")

        time.sleep(1800)

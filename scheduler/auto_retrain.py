"""
===========================================================
AUTO RETRAIN SCHEDULER
===========================================================
"""

import time

import config

from api.weather_api import download_weather_data
from preprocessing.preprocess import preprocess_dataset
from training.model_selection import select_best_model


# ===========================================================
# RETRAIN
# ===========================================================

def retrain():

    print("\n" + "=" * 60)
    print("AUTO RETRAIN STARTED")
    print("=" * 60)

    download_weather_data()

    preprocess_dataset()

    select_best_model()

    print("\nRetraining Completed.")


# ===========================================================
# MAIN LOOP
# ===========================================================

if __name__ == "__main__":

    while True:

        retrain()

        print(

            f"\nWaiting {config.RETRAIN_INTERVAL_DAYS} days..."

        )

        time.sleep(

            config.RETRAIN_INTERVAL_DAYS

            * 24

            * 60

            * 60

        )

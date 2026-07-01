"""
===========================================================
PREPROCESS NASA WEATHER DATASET
===========================================================
"""

import os
import numpy as np
import pandas as pd

import config


# ===========================================================
# PREPROCESS DATASET
# ===========================================================

def preprocess_dataset():

    print("\n" + "=" * 60)
    print("PREPROCESSING DATASET")
    print("=" * 60)

    # -------------------------------------------------------
    # CHECK DATASET
    # -------------------------------------------------------

    if not os.path.exists(config.DATASET_PATH):

        raise FileNotFoundError(
            f"{config.DATASET_PATH} not found."
        )

    # -------------------------------------------------------
    # LOAD DATASET
    # -------------------------------------------------------

    df = pd.read_csv(config.DATASET_PATH)

    print(f"\nOriginal Rows : {len(df)}")

    # -------------------------------------------------------
    # REMOVE MISSING VALUES
    # -------------------------------------------------------

    df.dropna(inplace=True)

    # -------------------------------------------------------
    # WIND DIRECTION
    # Degrees → Sin & Cos
    # -------------------------------------------------------

    radians = np.deg2rad(
        df["WIND DIRECTION AT 2M (IN DEGREES)"]
    )

    df["WIND_DIR_SIN"] = np.sin(radians)

    df["WIND_DIR_COS"] = np.cos(radians)

    # -------------------------------------------------------
    # PRESSURE
    # kPa → hPa
    # -------------------------------------------------------

    df["PRESSURE (hPa)"] = df["PRESSURE (kPa)"] * 10

    # -------------------------------------------------------
    # REMOVE OLD COLUMNS
    # -------------------------------------------------------

    df.drop(

        columns=[

            "WIND DIRECTION AT 2M (IN DEGREES)",

            "PRESSURE (kPa)"

        ],

        inplace=True

    )

    # -------------------------------------------------------
    # CREATE NEXT HOUR TARGETS
    # -------------------------------------------------------

    for column in config.TARGET_COLUMNS:

        df[column + "_NEXT"] = df[column].shift(-1)

    # -------------------------------------------------------
    # REMOVE LAST ROW
    # -------------------------------------------------------

    df.dropna(inplace=True)

    df.reset_index(

        drop=True,

        inplace=True

    )

    # -------------------------------------------------------
    # SAVE DATASET
    # -------------------------------------------------------

    df.to_csv(

        config.DATASET_PATH,

        index=False

    )

    print("\nPreprocessing Complete")

    print(f"Rows : {len(df)}")

    print(f"Columns : {len(df.columns)}")

    print(f"Saved : {config.DATASET_PATH}")

    print("\nFinal Columns\n")

    for column in df.columns:

        print(column)

    return df


# ===========================================================

if __name__ == "__main__":

    preprocess_dataset()

"""
===========================================================
NASA POWER WEATHER DATA DOWNLOAD
===========================================================
"""

import os
import json
import requests
import pandas as pd

from datetime import datetime, timedelta

import config

DATASET_META_PATH = os.path.join("data", "dataset_meta.json")
CACHE_MAX_AGE_HOURS = 24


def mark_dataset_ready(lat, lon):
    """Record that the on-disk dataset is downloaded AND preprocessed
    for these coordinates. Call only after preprocessing succeeds."""
    os.makedirs("data", exist_ok=True)
    with open(DATASET_META_PATH, "w") as f:
        json.dump({
            "latitude": lat,
            "longitude": lon,
            "downloaded_at": datetime.now().isoformat(),
        }, f)


def dataset_is_fresh(lat, lon):
    """Public: dataset on disk is preprocessed, for these coords, < 24h old."""
    return _cache_is_fresh(lat, lon)


def _cache_is_fresh(lat, lon):
    """True when the dataset on disk is for these coords and < 24h old."""
    if not (os.path.exists(DATASET_META_PATH) and os.path.exists(config.DATASET_PATH)):
        return False
    try:
        with open(DATASET_META_PATH) as f:
            meta = json.load(f)
        same_coords = (
            abs(meta["latitude"] - lat) < 1e-4 and
            abs(meta["longitude"] - lon) < 1e-4
        )
        age_hours = (
            datetime.now() - datetime.fromisoformat(meta["downloaded_at"])
        ).total_seconds() / 3600
        return same_coords and age_hours < CACHE_MAX_AGE_HOURS
    except Exception:
        return False


# ===========================================================
# DOWNLOAD NASA WEATHER DATA
# ===========================================================

def download_weather_data(latitude=None, longitude=None):

    print("\n" + "=" * 60)
    print("DOWNLOADING NASA POWER DATA")
    print("=" * 60)

    lat = latitude if latitude is not None else config.LATITUDE
    lon = longitude if longitude is not None else config.LONGITUDE

    print(f"Location : {lat:.4f}, {lon:.4f}")


    end_date = datetime.today()

    start_date = end_date - timedelta(
        days=365 * config.HISTORICAL_YEARS
    )

    start = start_date.strftime("%Y%m%d")
    end = end_date.strftime("%Y%m%d")

    params = {

        "parameters": ",".join(config.NASA_PARAMETERS),

        "community": "RE",

        "longitude": lon,

        "latitude": lat,

        "start": start,

        "end": end,

        "format": "JSON"

    }

    response = requests.get(

        config.NASA_POWER_BASE_URL,

        params=params,

        timeout=60

    )

    if response.status_code != 200:

        raise Exception(
            f"NASA API Error : {response.status_code}"
        )

    data = response.json()

    parameters = data["properties"]["parameter"]

    timestamps = list(parameters["T2M"].keys())

    dataset = []

    for ts in timestamps:

        dataset.append({

            "Timestamp": ts,

            "TEMPERATURE ( C)": parameters["T2M"][ts],

            "RELATIVE HUMIDITY (%)": parameters["RH2M"][ts],

            "WIND SPEED AT 2 M (M/S)": parameters["WS2M"][ts],

            "WIND DIRECTION AT 2M (IN DEGREES)": parameters["WD2M"][ts],

            "RAINFALL (mm)": parameters["PRECTOTCORR"][ts],

            "PRESSURE (kPa)": parameters["PS"][ts],

            "IRRADIANCE (MJ/hr)": parameters["ALLSKY_SFC_SW_DWN"][ts]

        })

    df = pd.DataFrame(dataset)

    os.makedirs(config.DATA_FOLDER, exist_ok=True)

    df.to_csv(

        config.DATASET_PATH,

        index=False

    )

    print("\nDownload Complete")

    print(f"Rows    : {len(df)}")

    print(f"Columns : {len(df.columns)}")

    print(f"Saved   : {config.DATASET_PATH}")

    return df


# ===========================================================

if __name__ == "__main__":

    download_weather_data()

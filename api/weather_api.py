"""
===========================================================
NASA POWER WEATHER DATA DOWNLOAD
===========================================================
"""

import os
import requests
import pandas as pd

from datetime import datetime, timedelta

import config

# ===========================================================
# DOWNLOAD NASA WEATHER DATA
# ===========================================================

def download_weather_data():

    print("\n" + "=" * 60)
    print("DOWNLOADING NASA POWER DATA")
    print("=" * 60)

    end_date = datetime.today()

    start_date = end_date - timedelta(
        days=365 * config.HISTORICAL_YEARS
    )

    start = start_date.strftime("%Y%m%d")
    end = end_date.strftime("%Y%m%d")

    params = {

        "parameters": ",".join(config.NASA_PARAMETERS),

        "community": "RE",

        "longitude": config.LONGITUDE,

        "latitude": config.LATITUDE,

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

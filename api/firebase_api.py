import firebase_admin
from datetime import datetime
from firebase_admin import credentials
from firebase_admin import db

# ==========================================================
# INITIALIZE FIREBASE
# ==========================================================

if not firebase_admin._apps:

    cred = credentials.Certificate(
        "serviceAccountKey.json"
    )

    firebase_admin.initialize_app(

        cred,

        {

            "databaseURL":

            "https://weather-app-2-920f0-default-rtdb.firebaseio.com/"

        }

    )

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

    sensor_data = {

        "temperature":

        read_value(

            "analytics/daily/avg_temperature"

        ),

        "humidity":

        read_value(

            "analytics/daily/avg_humidity"

        ),

        "wind_speed":

        read_value(

            "analytics/wind/average_speed"

        ),

        "wind_direction":

        read_value(

            "analytics/wind/average_direction"

        ),

        "rainfall":

        read_value(

            "analytics/rainfall/rainfall_percentage"

        ),

        "pressure":

        read_value(

            "analytics/daily/avg_pressure"

        ),

        "irradiance":

        read_value(

            "analytics/light/avg_light"

        )

    }

    return sensor_data

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

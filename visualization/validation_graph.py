"""
===========================================================
VALIDATION GRAPH
===========================================================
Actual vs Predicted Weather Parameters
===========================================================
"""

import firebase_admin
import matplotlib.pyplot as plt

from firebase_admin import credentials
from firebase_admin import db


# ===========================================================
# FIREBASE
# ===========================================================

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


# ===========================================================
# GRAPH
# ===========================================================

def plot_validation(parameter="temperature"):

    data = db.reference("validation").get()

    if not data:

        print("No validation data found.")

        return

    timestamps = []

    actual = []

    predicted = []

    for timestamp in sorted(data.keys()):

        item = data[timestamp]

        try:

            timestamps.append(timestamp)

            actual.append(

                item["actual"][parameter]

            )

            predicted.append(

                item["predicted"][parameter]

            )

        except KeyError:

            continue

    plt.figure(figsize=(12,6))

    plt.plot(

        timestamps,

        actual,

        marker="o",

        label="Actual"

    )

    plt.plot(

        timestamps,

        predicted,

        marker="s",

        label="Predicted"

    )

    plt.title(

        f"Actual vs Predicted {parameter.title()}"

    )

    plt.xlabel("Time")

    plt.ylabel(parameter.title())

    plt.xticks(rotation=45)

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.show()


# ===========================================================

if __name__ == "__main__":

    plot_validation("temperature")
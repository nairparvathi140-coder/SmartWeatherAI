"""
===========================================================
SMART WEATHER AI CONFIGURATION
===========================================================
"""

# ===========================================================
# WEATHER STATION LOCATION
# ===========================================================

LATITUDE = 12.9716
LONGITUDE = 77.5946

# ===========================================================
# NASA POWER API
# ===========================================================

NASA_POWER_BASE_URL = (
    "https://power.larc.nasa.gov/api/temporal/hourly/point"
)

HISTORICAL_YEARS = 2

NASA_PARAMETERS = [

    "T2M",             # Temperature (°C)

    "RH2M",            # Relative Humidity (%)

    "WS2M",            # Wind Speed (m/s)

    "WD2M",            # Wind Direction (degrees)

    "PRECTOTCORR",     # Rainfall (mm)

    "PS",              # Surface Pressure (kPa)

    "ALLSKY_SFC_SW_DWN"

]

# ===========================================================
# MACHINE LEARNING MODELS
# ===========================================================

MODELS = [

    "Linear Regression",

    "Random Forest",

    "Extra Trees",

    "XGBoost"

]

N_ESTIMATORS = 200

RANDOM_STATE = 42

TEST_SIZE = 0.20

# ===========================================================
# FILE PATHS
# ===========================================================

DATA_FOLDER = "data"

MODEL_FOLDER = "models"

LOG_FOLDER = "logs"

DATASET_PATH = "data/weather_dataset.csv"

MODEL_PATH = "models/best_model.pkl"

MODEL_INFO_PATH = "models/model_info.json"

MODEL_RESULTS_PATH = "logs/model_comparison.csv"

FEATURE_IMPORTANCE_PATH = "logs/feature_importance.csv"

VALIDATION_LOG_PATH = "logs/validation_log.csv"

# ===========================================================
# DATASET COLUMNS
# ===========================================================

FEATURE_COLUMNS = [

    "TEMPERATURE ( C)",

    "RELATIVE HUMIDITY (%)",

    "WIND SPEED AT 2 M (M/S)",

    "WIND_DIR_SIN",

    "WIND_DIR_COS",

    "RAINFALL (mm)",

    "PRESSURE (hPa)",

    "IRRADIANCE (MJ/hr)"

]

TARGET_COLUMNS = [

    "TEMPERATURE ( C)",

    "RELATIVE HUMIDITY (%)",

    "WIND SPEED AT 2 M (M/S)",

    "WIND_DIR_SIN",

    "WIND_DIR_COS",

    "RAINFALL (mm)",

    "PRESSURE (hPa)",

    "IRRADIANCE (MJ/hr)"

]

# ===========================================================
# FIREBASE
# ===========================================================

SERVICE_ACCOUNT_FILE = "serviceAccountKey.json"

# ===========================================================
# PREDICTION
# ===========================================================

PREDICTION_INTERVAL_MINUTES = 30

# ===========================================================
# RETRAINING
# ===========================================================

RETRAIN_INTERVAL_DAYS = 7

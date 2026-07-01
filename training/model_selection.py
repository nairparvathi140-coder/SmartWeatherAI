"""
===========================================================
MODEL SELECTION & TRAINING
===========================================================
Automatically trains multiple ML models and selects the best.
===========================================================
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from datetime import datetime

import config

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor
)

from xgboost import XGBRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# ===========================================================
# MODEL SELECTION
# ===========================================================

def select_best_model():

    print("\n" + "=" * 60)
    print("MODEL SELECTION")
    print("=" * 60)

    # -------------------------------------------------------
    # CREATE REQUIRED FOLDERS
    # -------------------------------------------------------

    os.makedirs(config.MODEL_FOLDER, exist_ok=True)
    os.makedirs(config.LOG_FOLDER, exist_ok=True)

    # -------------------------------------------------------
    # LOAD DATASET
    # -------------------------------------------------------

    if not os.path.exists(config.DATASET_PATH):

        raise FileNotFoundError(
            f"{config.DATASET_PATH} not found."
        )

    df = pd.read_csv(config.DATASET_PATH)

    X = df[config.FEATURE_COLUMNS]

    Y = df[
        [col + "_NEXT" for col in config.TARGET_COLUMNS]
    ]

    # -------------------------------------------------------
    # TRAIN / TEST SPLIT
    # -------------------------------------------------------

    X_train, X_test, Y_train, Y_test = train_test_split(

        X,

        Y,

        test_size=config.TEST_SIZE,

        random_state=config.RANDOM_STATE,

        shuffle=True

    )

    print(f"\nTraining Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")

    # -------------------------------------------------------
    # MODELS
    # -------------------------------------------------------

    models = {

        "Linear Regression":

        MultiOutputRegressor(

            LinearRegression()

        ),

        "Random Forest":

        MultiOutputRegressor(

            RandomForestRegressor(

                n_estimators=config.N_ESTIMATORS,

                random_state=config.RANDOM_STATE,

                n_jobs=-1

            )

        ),

        "Extra Trees":

        MultiOutputRegressor(

            ExtraTreesRegressor(

                n_estimators=config.N_ESTIMATORS,

                random_state=config.RANDOM_STATE,

                n_jobs=-1

            )

        ),

        "XGBoost":

        MultiOutputRegressor(

            XGBRegressor(

                n_estimators=config.N_ESTIMATORS,

                max_depth=6,

                learning_rate=0.1,

                objective="reg:squarederror",

                random_state=config.RANDOM_STATE,

                n_jobs=-1

            )

        )

    }

    # -------------------------------------------------------
    # VARIABLES
    # -------------------------------------------------------

    results = []

    best_model = None

    best_name = ""

    best_r2 = -999

    best_rmse = 999

    best_mae = 999

    best_mse = 999
    # -------------------------------------------------------
    # TRAIN ALL MODELS
    # -------------------------------------------------------

    for name, model in models.items():

        print("\n" + "-" * 60)
        print(f"Training : {name}")
        print("-" * 60)

        # ---------------------------------------------------
        # TRAIN MODEL
        # ---------------------------------------------------

        model.fit(

            X_train,

            Y_train

        )

        # ---------------------------------------------------
        # PREDICTIONS
        # ---------------------------------------------------

        train_prediction = model.predict(

            X_train

        )

        test_prediction = model.predict(

            X_test

        )

        # ---------------------------------------------------
        # METRICS
        # ---------------------------------------------------

        train_r2 = r2_score(

            Y_train,

            train_prediction

        )

        test_r2 = r2_score(

            Y_test,

            test_prediction

        )

        mae = mean_absolute_error(

            Y_test,

            test_prediction

        )

        mse = mean_squared_error(

            Y_test,

            test_prediction

        )

        rmse = np.sqrt(

            mse

        )

        # ---------------------------------------------------
        # DISPLAY RESULTS
        # ---------------------------------------------------

        print(f"Train R² : {train_r2:.4f}")

        print(f"Test  R² : {test_r2:.4f}")

        print(f"MAE      : {mae:.4f}")

        print(f"MSE      : {mse:.4f}")

        print(f"RMSE     : {rmse:.4f}")

        # ---------------------------------------------------
        # SAVE RESULTS
        # ---------------------------------------------------

        results.append({

            "Model": name,

            "Train R2": train_r2,

            "Test R2": test_r2,

            "MAE": mae,

            "MSE": mse,

            "RMSE": rmse

        })

        # ---------------------------------------------------
        # BEST MODEL SELECTION
        # Priority:
        # 1. Highest Test R²
        # 2. Lowest RMSE
        # 3. Lowest MAE
        # ---------------------------------------------------

        better = False

        if test_r2 > best_r2:

            better = True

        elif abs(test_r2 - best_r2) < 0.005:

            if rmse < best_rmse:

                better = True

            elif abs(rmse - best_rmse) < 1e-6:

                if mae < best_mae:

                    better = True

        if better:

            best_model = model

            best_name = name

            best_r2 = test_r2

            best_mae = mae

            best_mse = mse

            best_rmse = rmse

    # -------------------------------------------------------
    # MODEL COMPARISON TABLE
    # -------------------------------------------------------

    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(

        by="Test R2",

        ascending=False

    )

    comparison.to_csv(

        config.MODEL_RESULTS_PATH,

        index=False

    )

    print("\n")
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(comparison)
    # -------------------------------------------------------
    # SAVE BEST MODEL
    # -------------------------------------------------------

    joblib.dump(

        best_model,

        config.MODEL_PATH

    )

    # -------------------------------------------------------
    # FEATURE IMPORTANCE
    # (Tree Models Only)
    # -------------------------------------------------------

    if best_name in ["Random Forest", "Extra Trees"]:

        importance = best_model.estimators_[0].feature_importances_

        feature_df = pd.DataFrame({

            "Feature": config.FEATURE_COLUMNS,

            "Importance": importance

        })

        feature_df = feature_df.sort_values(

            by="Importance",

            ascending=False

        )

        feature_df.to_csv(

            config.FEATURE_IMPORTANCE_PATH,

            index=False

        )

    # -------------------------------------------------------
    # SAVE MODEL INFORMATION
    # -------------------------------------------------------

    model_info = {

        "best_model": best_name,

        "r2": float(best_r2),

        "mae": float(best_mae),

        "mse": float(best_mse),

        "rmse": float(best_rmse),

        "trained_location": {

            "latitude": config.LATITUDE,

            "longitude": config.LONGITUDE

        },

        "training_date": datetime.now().strftime(

            "%Y-%m-%d %H:%M:%S"

        ),

        "features": config.FEATURE_COLUMNS,

        "targets": config.TARGET_COLUMNS

    }

    with open(

        config.MODEL_INFO_PATH,

        "w"

    ) as file:

        json.dump(

            model_info,

            file,

            indent=4

        )

    # -------------------------------------------------------
    # PRINT SUMMARY
    # -------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("BEST MODEL SELECTED")
    print("=" * 60)

    print(f"Model : {best_name}")
    print(f"R²    : {best_r2:.4f}")
    print(f"MAE   : {best_mae:.4f}")
    print(f"MSE   : {best_mse:.4f}")
    print(f"RMSE  : {best_rmse:.4f}")

    print("\nFiles Saved")
    print("-" * 60)

    print("Best Model          :", config.MODEL_PATH)
    print("Model Info          :", config.MODEL_INFO_PATH)
    print("Comparison Report   :", config.MODEL_RESULTS_PATH)

    if best_name in ["Random Forest", "Extra Trees"]:

        print("Feature Importance  :", config.FEATURE_IMPORTANCE_PATH)

    print("=" * 60)

    return best_model


# ===========================================================
# MAIN
# ===========================================================

if __name__ == "__main__":

    select_best_model()   

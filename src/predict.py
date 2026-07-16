"""
===========================================================
Project : AI-Based Cyber Threat Detection Framework
File    : predict.py

Author  : Chandrakant

Purpose :
Predict network traffic attack using trained ML model.
===========================================================
"""


from pathlib import Path
import joblib
import pandas as pd
import numpy as np


# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent


MODEL_DIR = BASE_DIR / "models"


# ==========================================================
# LOAD TRAINED FILES
# ==========================================================


print("=" * 60)
print("Loading Trained Model...")
print("=" * 60)


MODEL_PATH = MODEL_DIR / "best_model.pkl"

SCALER_PATH = MODEL_DIR / "scaler.pkl"

ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"



model = joblib.load(
    MODEL_PATH
)


scaler = joblib.load(
    SCALER_PATH
)


encoder = joblib.load(
    ENCODER_PATH
)


print("Model Loaded Successfully.")
print("Scaler Loaded Successfully.")
print("Encoder Loaded Successfully.")

# ==========================================================
# PREPARE INPUT DATA
# ==========================================================


def prepare_input(file_path):

    print("\nLoading Input Data...")

    df = pd.read_csv(file_path)


    print("Input Shape :", df.shape)


    # Remove extra spaces from columns

    df.columns = df.columns.str.strip()


    # Remove Label column if present

    if "Label" in df.columns:

        df = df.drop(
            columns=["Label"]
        )


    # Replace infinite values

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )


    # Handle missing values

    df.fillna(
        0,
        inplace=True
    )


    return df



# ==========================================================
# PREDICTION FUNCTION
# ==========================================================


def predict_attack(file_path):


    # Prepare Data

    data = prepare_input(
        file_path
    )


    print("\nScaling Input Data...")


    scaled_data = scaler.transform(
        data
    )


    print("Making Prediction...")


    prediction = model.predict(
        scaled_data
    )


    # Convert number to attack name

    attack_name = encoder.inverse_transform(
        prediction
    )


    return attack_name

# ==========================================================
# MAIN TESTING
# ==========================================================


if __name__ == "__main__":


    print("\n" + "=" * 60)
    print("AI CYBER THREAT PREDICTION SYSTEM")
    print("=" * 60)


    # ======================================================
    # GIVE TEST CSV FILE PATH HERE
    # ======================================================


    TEST_FILE = (
        BASE_DIR
        /
        "data"
        /
        "processed"
        /
        "CICIDS2017_FeatureSelected.csv"
    )



    try:


        prediction = predict_attack(
            TEST_FILE
        )


        print("\n" + "=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)



        unique, counts = np.unique(
            prediction,
            return_counts=True
        )


        for attack, count in zip(
            unique,
            counts
        ):

            print(
                f"{attack} : {count} samples"
            )



        print("\nPrediction Completed Successfully.")



    except Exception as e:


        print("\nPrediction Failed")

        print(
            "Error :",
            e
        )
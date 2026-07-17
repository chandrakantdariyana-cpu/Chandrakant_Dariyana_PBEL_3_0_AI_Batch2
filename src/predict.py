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

from logger import logger


# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"


# ==========================================================
# LOAD TRAINED FILES
# ==========================================================

logger.info("Prediction module started.")

print("=" * 60)
print("Loading Trained Model...")
print("=" * 60)

MODEL_PATH = MODEL_DIR / "best_model.pkl"

SCALER_PATH = MODEL_DIR / "scaler.pkl"

ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

logger.info(f"Loading model from: {MODEL_PATH}")

model = joblib.load(
    MODEL_PATH
)

logger.info(f"Loading scaler from: {SCALER_PATH}")

scaler = joblib.load(
    SCALER_PATH
)

logger.info(f"Loading label encoder from: {ENCODER_PATH}")

encoder = joblib.load(
    ENCODER_PATH
)

logger.info("All trained files loaded successfully.")

print("Model Loaded Successfully.")
print("Scaler Loaded Successfully.")
print("Encoder Loaded Successfully.")


# ==========================================================
# PREPARE INPUT DATA
# ==========================================================

def prepare_input(file_path):

    logger.info(f"Preparing input data from: {file_path}")

    print("\nLoading Input Data...")

    df = pd.read_csv(file_path)

    logger.info("Input dataset loaded successfully.")
    logger.debug(f"Input dataset shape: {df.shape}")

    print("Input Shape :", df.shape)

    df.columns = df.columns.str.strip()

    logger.info("Column names stripped successfully.")

    if "Label" in df.columns:

        logger.info("Label column found and removed.")

        df = df.drop(
            columns=["Label"]
        )

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    logger.info("Infinite values replaced with NaN.")

    missing_values = df.isna().sum().sum()

    logger.info(f"Missing values found: {missing_values}")

    df.fillna(
        0,
        inplace=True
    )

    logger.info("Missing values filled with 0.")
    logger.debug(f"Prepared input shape: {df.shape}")

    return df


# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict_attack(file_path):

    logger.info("Prediction process started.")

    data = prepare_input(
        file_path
    )

    print("\nScaling Input Data...")

    logger.info("Scaling input data.")

    scaled_data = scaler.transform(
        data
    )

    logger.info("Input data scaled successfully.")
    logger.debug(f"Scaled data shape: {scaled_data.shape}")

    print("Making Prediction...")

    logger.info("Generating predictions.")

    prediction = model.predict(
        scaled_data
    )

    logger.info("Predictions generated successfully.")
    logger.debug(f"Total predictions: {len(prediction)}")

    attack_name = encoder.inverse_transform(
        prediction
    )

    logger.info("Prediction labels decoded successfully.")

    return attack_name


# ==========================================================
# MAIN TESTING
# ==========================================================

def main():
    logger.info("Executing predict.py")

    print("\n" + "=" * 60)
    print("AI CYBER THREAT PREDICTION SYSTEM")
    print("=" * 60)

    TEST_FILE = (
            BASE_DIR
            /
            "data"
            /
            "processed"
            /
            "CICIDS2017_FeatureSelected.csv"
    )

    logger.info(f"Test file selected: {TEST_FILE}")

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

        logger.info("Prediction summary generated.")

        for attack, count in zip(
                unique,
                counts
        ):
            logger.debug(f"{attack}: {count} samples")

            print(
                f"{attack} : {count} samples"
            )

        print("\nPrediction Completed Successfully.")

        logger.info("Prediction process completed successfully.")

    except Exception as e:

        logger.exception(f"Prediction failed: {e}")

        print("\nPrediction Failed")

        print(
            "Error :",
            e
        )

if __name__ == "__main__":
    main()

"""
===========================================================
Project : AI-Based Cyber Threat Detection Framework
File    : train_model.py
Author  : Chandrakant
Purpose : Prepare dataset for Machine Learning model training.
===========================================================
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "CICIDS2017_FeatureSelected.csv"


# ==========================================================
# Data Preparation Function
# ==========================================================

def prepare_dataset():

    try:

        print("=" * 60)
        print("Loading Feature Selected Dataset...")

        df = pd.read_csv(DATA_PATH)

        print("Dataset loaded successfully.")
        print(f"Dataset Shape : {df.shape}")

        # Remove extra spaces from column names
        df.columns = df.columns.str.strip()

        # Check Label column
        if "Label" not in df.columns:
            raise ValueError("'Label' column not found.")

        # ---------------------------------------------
        # Split Features and Target
        # ---------------------------------------------

        X = df.drop(columns=["Label"])
        y = df["Label"]

        print(f"\nNumber of Features : {X.shape[1]}")
        print(f"Number of Samples  : {X.shape[0]}")

        # ---------------------------------------------
        # Encode Labels
        # ---------------------------------------------

        print("\nEncoding Labels...")

        label_encoder = LabelEncoder()

        y_encoded = label_encoder.fit_transform(y)

        print("Encoding completed.")

        print(f"Total Classes : {len(label_encoder.classes_)}")
        print("\nClasses Found:")

        for index, label in enumerate(label_encoder.classes_):
            print(f"{index} -> {label}")

        # ---------------------------------------------
        # Train Test Split
        # ---------------------------------------------

        print("\nSplitting Dataset...")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_encoded,
            test_size=0.20,
            random_state=42,
            stratify=y_encoded
        )

        print("Dataset split completed.\n")

        print(f"Training Samples : {X_train.shape[0]}")
        print(f"Testing Samples  : {X_test.shape[0]}")

        print("=" * 60)
        print("Dataset is Ready for Machine Learning.")
        print("=" * 60)

        return X_train, X_test, y_train, y_test, label_encoder

    except FileNotFoundError:

        print("ERROR : Feature Selected Dataset not found.")
        print(DATA_PATH)

    except Exception as error:

        print(f"Unexpected Error : {error}")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    prepare_dataset()
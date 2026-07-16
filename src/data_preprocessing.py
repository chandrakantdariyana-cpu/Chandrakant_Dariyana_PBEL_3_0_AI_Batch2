"""
===========================================================
Project : AI-Based Cyber Threat Detection Framework
File    : data_preprocessing.py

Author  : Chandrakant

Purpose :
Data preprocessing pipeline for Machine Learning models.
===========================================================
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "CICIDS2017_FeatureSelected.csv"
)

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset():

    print("=" * 60)
    print("Loading Feature Selected Dataset...")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)

    # Development Sample
    df = df.sample(
        n=200000,
        random_state=42
    )

    print(f"Dataset Shape : {df.shape}")

    return df


# ==========================================================
# CLEAN DATASET
# ==========================================================

def clean_dataset(df):

    print("\nCleaning Dataset...")

    # Remove extra spaces
    df.columns = df.columns.str.strip()

    # Remove duplicate rows
    duplicate_rows = df.duplicated().sum()

    df.drop_duplicates(inplace=True)

    # Replace Infinite Values
    inf_count = np.isinf(
        df.select_dtypes(include=[np.number])
    ).sum().sum()

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    # Missing Values
    missing_values = df.isna().sum().sum()

    df.dropna(inplace=True)

    print(f"Duplicate Rows Removed : {duplicate_rows}")
    print(f"Infinite Values Found  : {inf_count}")
    print(f"Missing Values Removed : {missing_values}")
    print(f"Final Shape            : {df.shape}")

    return df


# ==========================================================
# SPLIT FEATURES & LABEL
# ==========================================================

def split_features_labels(df):

    X = df.drop(
        columns=["Label"]
    )

    y = df["Label"]

    print(f"\nFeatures : {X.shape[1]}")
    print(f"Samples  : {X.shape[0]}")

    return X, y


# ==========================================================
# LABEL ENCODING
# ==========================================================

def encode_labels(y):

    print("\nEncoding Labels...")

    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(y)

    joblib.dump(
        encoder,
        MODEL_DIR / "label_encoder.pkl"
    )

    return y_encoded, encoder

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

def split_dataset(X, y):

    print("\nSplitting Dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
        shuffle=True
    )

    print(f"Training Samples : {X_train.shape[0]}")
    print(f"Testing Samples  : {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test


# ==========================================================
# FEATURE SCALING
# ==========================================================

def scale_features(X_train, X_test):

    print("\nScaling Features...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    # Save Scaler
    joblib.dump(
        scaler,
        MODEL_DIR / "scaler.pkl"
    )

    return X_train_scaled, X_test_scaled, scaler


# ==========================================================
# COMPLETE DATA PREPARATION
# ==========================================================

def prepare_dataset():

    df = load_dataset()

    df = clean_dataset(df)

    X, y = split_features_labels(df)

    y_encoded, encoder = encode_labels(y)

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y_encoded
    )

    X_train, X_test, scaler = scale_features(
        X_train,
        X_test
    )

    print("\nDataset Ready For Training.")
    print("=" * 60)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        encoder,
        scaler
    )

# ==========================================================
# MAIN (TESTING)
# ==========================================================

if __name__ == "__main__":

    try:

        (
            X_train,
            X_test,
            y_train,
            y_test,
            encoder,
            scaler

        ) = prepare_dataset()

        print("\n" + "=" * 60)
        print("DATA PREPROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 60)

        print(f"Training Data Shape : {X_train.shape}")
        print(f"Testing Data Shape  : {X_test.shape}")

        print(f"Training Labels : {len(y_train)}")
        print(f"Testing Labels  : {len(y_test)}")

        print("\nSaved Files")
        print("-" * 60)
        print(MODEL_DIR / "label_encoder.pkl")
        print(MODEL_DIR / "scaler.pkl")

    except Exception as e:

        print("\nError During Data Preprocessing")
        print(e)
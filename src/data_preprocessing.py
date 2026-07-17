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

from logger import logger


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

    logger.info("Loading feature selected dataset started.")

    print("=" * 60)
    print("Loading Feature Selected Dataset...")
    print("=" * 60)

    logger.info(f"Dataset path: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    logger.info("Dataset loaded successfully.")
    logger.debug(f"Original dataset shape: {df.shape}")

    df = df.sample(
        n=200000,
        random_state=42
    )

    logger.info("Development sample selected.")
    logger.debug(f"Sampled dataset shape: {df.shape}")

    print(f"Dataset Shape : {df.shape}")

    return df


# ==========================================================
# CLEAN DATASET
# ==========================================================

def clean_dataset(df):

    logger.info("Dataset cleaning started.")

    print("\nCleaning Dataset...")

    df.columns = df.columns.str.strip()

    logger.info("Column names stripped.")

    duplicate_rows = df.duplicated().sum()

    logger.info(f"Duplicate rows found: {duplicate_rows}")

    df.drop_duplicates(inplace=True)

    logger.debug(f"Shape after duplicate removal: {df.shape}")

    inf_count = np.isinf(
        df.select_dtypes(include=[np.number])
    ).sum().sum()

    logger.info(f"Infinite values found: {inf_count}")

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    logger.info("Infinite values replaced with NaN.")

    missing_values = df.isna().sum().sum()

    logger.info(f"Missing values found: {missing_values}")

    df.dropna(inplace=True)

    logger.debug(f"Shape after removing missing values: {df.shape}")

    print(f"Duplicate Rows Removed : {duplicate_rows}")
    print(f"Infinite Values Found  : {inf_count}")
    print(f"Missing Values Removed : {missing_values}")
    print(f"Final Shape            : {df.shape}")

    logger.info("Dataset cleaning completed.")

    return df


# ==========================================================
# SPLIT FEATURES & LABEL
# ==========================================================

def split_features_labels(df):

    logger.info("Splitting features and labels.")

    X = df.drop(
        columns=["Label"]
    )

    y = df["Label"]

    logger.debug(f"Feature shape: {X.shape}")
    logger.debug(f"Label shape: {y.shape}")

    print(f"\nFeatures : {X.shape[1]}")
    print(f"Samples  : {X.shape[0]}")

    return X, y


# ==========================================================
# LABEL ENCODING
# ==========================================================

def encode_labels(y):

    logger.info("Label encoding started.")

    print("\nEncoding Labels...")

    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(y)

    logger.info(f"Total classes encoded: {len(encoder.classes_)}")
    logger.debug(f"Classes: {list(encoder.classes_)}")

    joblib.dump(
        encoder,
        MODEL_DIR / "label_encoder.pkl"
    )

    logger.info(f"Label encoder saved at: {MODEL_DIR / 'label_encoder.pkl'}")

    return y_encoded, encoder


# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

def split_dataset(X, y):

    logger.info("Splitting dataset into training and testing.")

    print("\nSplitting Dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
        shuffle=True
    )

    logger.debug(f"Training feature shape: {X_train.shape}")
    logger.debug(f"Testing feature shape: {X_test.shape}")
    logger.debug(f"Training label count: {len(y_train)}")
    logger.debug(f"Testing label count: {len(y_test)}")

    print(f"Training Samples : {X_train.shape[0]}")
    print(f"Testing Samples  : {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test


# ==========================================================
# FEATURE SCALING
# ==========================================================

def scale_features(X_train, X_test):

    logger.info("Feature scaling started.")

    print("\nScaling Features...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    logger.info("Training features scaled.")

    X_test_scaled = scaler.transform(X_test)

    logger.info("Testing features scaled.")

    joblib.dump(
        scaler,
        MODEL_DIR / "scaler.pkl"
    )

    logger.info(f"Scaler saved at: {MODEL_DIR / 'scaler.pkl'}")

    return X_train_scaled, X_test_scaled, scaler


# ==========================================================
# COMPLETE DATA PREPARATION
# ==========================================================

def prepare_dataset():

    logger.info("Complete data preparation pipeline started.")

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

    logger.info("Dataset preparation completed successfully.")

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

    logger.info("Executing data_preprocessing.py")

    try:

        (
            X_train,
            X_test,
            y_train,
            y_test,
            encoder,
            scaler

        ) = prepare_dataset()

        logger.info("Data preprocessing executed successfully.")

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

        logger.exception(f"Error during data preprocessing: {e}")

        print("\nError During Data Preprocessing")
        print(e)
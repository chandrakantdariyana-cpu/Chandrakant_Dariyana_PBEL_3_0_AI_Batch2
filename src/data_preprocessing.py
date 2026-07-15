"""
===========================================================
Project : AI-Based Cyber Threat Detection Framework
File    : data_preprocessing.py

Author  : Chandrakant
Purpose : Data preprocessing and ML model training.
===========================================================
"""


from pathlib import Path
import pandas as pd
import joblib


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent


DATA_PATH = (
    BASE_DIR /
    "data" /
    "processed" /
    "CICIDS2017_FeatureSelected.csv"
)


MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)



# ==========================================================
# Data Preparation
# ==========================================================

def prepare_dataset():

    try:

        print("=" * 60)
        print("Loading Feature Selected Dataset...")
        print("=" * 60)


        # Load Dataset
        df = pd.read_csv(DATA_PATH)


        # Development Sample
        df = df.sample(
            n=200000,
            random_state=42
        )


        print("Initial Shape :", df.shape)



        # ==================================================
        # Data Cleaning
        # ==================================================

        print("\nCleaning Dataset...")


        # Remove spaces from column names
        df.columns = df.columns.str.strip()


        # Remove duplicate rows
        df.drop_duplicates(
            inplace=True
        )


        # Replace infinity values
        df.replace(
            [float("inf"), float("-inf")],
            pd.NA,
            inplace=True
        )


        # Remove missing values
        df.dropna(
            inplace=True
        )


        print(
            "After Cleaning :",
            df.shape
        )



        # ==================================================
        # Feature and Label Separation
        # ==================================================

        X = df.drop(
            columns=["Label"]
        )


        y = df["Label"]



        print(
            "Features :",
            X.shape[1]
        )

        print(
            "Samples :",
            X.shape[0]
        )



        # ==================================================
        # Label Encoding
        # ==================================================

        print("\nEncoding Labels...")


        encoder = LabelEncoder()


        y = encoder.fit_transform(y)



        # ==================================================
        # Train Test Split
        # ==================================================

        X_train, X_test, y_train, y_test = train_test_split(

            X,
            y,

            test_size=0.20,

            random_state=42,

            stratify=y
        )



        # ==================================================
        # Feature Scaling
        # ==================================================

        print("\nScaling Features...")


        scaler = StandardScaler()


        X_train = scaler.fit_transform(
            X_train
        )


        X_test = scaler.transform(
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


    except Exception as e:

        print(
            "Error :",
            e
        )





# ==========================================================
# Logistic Regression
# ==========================================================

def train_logistic_regression(
        X_train,
        X_test,
        y_train,
        y_test):


    print("\nTraining Logistic Regression...")


    model = LogisticRegression(

        max_iter=1000,

        random_state=42,

        n_jobs=-1
    )


    model.fit(
        X_train,
        y_train
    )



    predictions = model.predict(
        X_test
    )



    accuracy = accuracy_score(
        y_test,
        predictions
    )


    print(
        f"Accuracy : {accuracy:.4f}"
    )


    return model





# ==========================================================
# Random Forest
# ==========================================================

def train_random_forest(
        X_train,
        X_test,
        y_train,
        y_test):


    print("\nTraining Random Forest...")


    model = RandomForestClassifier(

        n_estimators=100,

        random_state=42,

        n_jobs=-1
    )


    model.fit(
        X_train,
        y_train
    )


    predictions = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    print(
        f"Accuracy : {accuracy:.4f}"
    )


    return model





# ==========================================================
# Save Model
# ==========================================================

def save_model(
        model,
        encoder,
        scaler,
        model_name):


    joblib.dump(

        model,

        MODEL_DIR /
        f"{model_name}.pkl"

    )


    joblib.dump(

        encoder,

        MODEL_DIR /
        "label_encoder.pkl"

    )


    joblib.dump(

        scaler,

        MODEL_DIR /
        "scaler.pkl"

    )


    print("\nModel Saved Successfully.")

    print(
        MODEL_DIR /
        f"{model_name}.pkl"
    )





# ==========================================================
# Main
# ==========================================================


if __name__ == "__main__":


    (
        X_train,
        X_test,
        y_train,
        y_test,
        encoder,
        scaler

    ) = prepare_dataset()



    # Train Logistic Regression

    model = train_logistic_regression(

        X_train,

        X_test,

        y_train,

        y_test

    )



    # Save Model

    save_model(

        model,

        encoder,

        scaler,

        "logistic_regression"

    )
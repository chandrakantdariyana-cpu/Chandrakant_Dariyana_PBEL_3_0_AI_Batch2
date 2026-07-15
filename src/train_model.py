"""
===========================================================
Project : AI-Based Cyber Threat Detection Framework
File    : train_model.py
Author  : Chandrakant
Purpose : Prepare dataset and train Logistic Regression model.
===========================================================
"""

from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score


# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "CICIDS2017_FeatureSelected.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ==========================================================
# Data Preparation
# ==========================================================

def prepare_dataset():

    try:

        print("=" * 60)
        print("Loading Feature Selected Dataset...")

        df = pd.read_csv(DATA_PATH)

        # Development sample (fast training)
        df = df.sample(n=200000, random_state=42)

        print(f"Development Sample Shape : {df.shape}")

        df.columns = df.columns.str.strip()

        X = df.drop(columns=["Label"])
        y = df["Label"]

        print(f"Features : {X.shape[1]}")
        print(f"Samples  : {X.shape[0]}")

        # Encode Labels
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)

        # Train Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        print("Dataset Ready for Machine Learning.")
        print("=" * 60)

        return X_train, X_test, y_train, y_test, encoder

    except Exception as e:

        print("Error :", e)


# ==========================================================
# Logistic Regression
# ==========================================================

def train_logistic_regression(X_train, X_test, y_train, y_test):

    print("\nTraining Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy : {accuracy:.4f}")

    return model


# ==========================================================
# Random Forest
# ==========================================================

def train_random_forest(X_train, X_test, y_train, y_test):

    print("\n" + "=" * 60)
    print("Training Random Forest...")
    print("=" * 60)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy : {accuracy:.4f}")

    return model


# ==========================================================
# Save Model
# ==========================================================

def save_model(model, encoder, model_name):

    joblib.dump(
        model,
        MODEL_DIR / f"{model_name}.pkl"
    )

    joblib.dump(
        encoder,
        MODEL_DIR / "label_encoder.pkl"
    )

    print("\nModel Saved Successfully.")
    print(MODEL_DIR / f"{model_name}.pkl")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    X_train, X_test, y_train, y_test, encoder = prepare_dataset()

    model = train_logistic_regression(
        X_train,
        X_test,
        y_train,
        y_test
    )

    save_model(
        model,
        encoder,
        "logistic_regression"
    )
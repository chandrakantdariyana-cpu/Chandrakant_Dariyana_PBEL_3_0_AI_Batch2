"""
===========================================================
Project : AI-Based Cyber Threat Detection Framework
File    : train_all_models.py
Author  : Chandrakant
Purpose : Train and compare multiple Machine Learning models.
===========================================================
"""

from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "CICIDS2017_FeatureSelected.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# ==========================================================
# Dataset Preparation
# ==========================================================

def prepare_dataset():

    print("=" * 60)
    print("Loading Feature Selected Dataset...")

    df = pd.read_csv(DATA_PATH)

    # Development Sample (Fast Training)
    df = df.sample(n=200000, random_state=42)

    print(f"Development Sample Shape : {df.shape}")

    df.columns = df.columns.str.strip()

    X = df.drop(columns=["Label"])
    y = df["Label"]

    print(f"Features : {X.shape[1]}")
    print(f"Samples  : {X.shape[0]}")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Dataset Ready.")
    print("=" * 60)

    return X_train, X_test, y_train, y_test, encoder

# ==========================================================
# Logistic Regression
# ==========================================================

def train_logistic_regression(X_train, X_test, y_train, y_test):

    print("\n" + "=" * 60)
    print("Training Logistic Regression...")
    print("=" * 60)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    return model, accuracy, precision, recall, f1

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

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    return model, accuracy, precision, recall, f1

# ==========================================================
# Decision Tree
# ==========================================================

def train_decision_tree(X_train, X_test, y_train, y_test):

    print("\n" + "=" * 60)
    print("Training Decision Tree...")
    print("=" * 60)

    model = DecisionTreeClassifier(
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    return model, accuracy, precision, recall, f1

# ==========================================================
# Generic Model Evaluation
# ==========================================================

def evaluate_model(model, model_name, X_train, X_test, y_train, y_test):

    print("\n" + "=" * 60)
    print(f"Training {model_name}...")
    print("=" * 60)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Trained_Model": model
    }

# ==========================================================
# Save Model
# ==========================================================

def save_model(model, model_name):

    model_path = MODEL_DIR / f"{model_name}.pkl"

    joblib.dump(model, model_path)

    print(f"\n{model_name} saved successfully.")
    print(model_path)

# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    X_train, X_test, y_train, y_test, encoder = prepare_dataset()

    results = []

    # Logistic Regression
    result = evaluate_model(
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "Logistic Regression",
        X_train,
        X_test,
        y_train,
        y_test
    )

    results.append(result)

    save_model(
        result["Trained_Model"],
        "logistic_regression"
    )

    # Random Forest
    result = evaluate_model(
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        "Random Forest",
        X_train,
        X_test,
        y_train,
        y_test
    )

    results.append(result)

    save_model(
        result["Trained_Model"],
        "random_forest"
    )

    # Decision Tree
    result = evaluate_model(
        DecisionTreeClassifier(
            random_state=42
        ),
        "Decision Tree",
        X_train,
        X_test,
        y_train,
        y_test
    )

    results.append(result)

    save_model(
        result["Trained_Model"],
        "decision_tree"
    )

    # K-Nearest Neighbors
    result = evaluate_model(
        KNeighborsClassifier(
            n_neighbors=5
        ),
        "KNN",
        X_train,
        X_test,
        y_train,
        y_test
    )

    results.append(result)

    save_model(
        result["Trained_Model"],
        "knn"
    )

    # Gaussian Naive Bayes
    result = evaluate_model(
        GaussianNB(),
        "Naive Bayes",
        X_train,
        X_test,
        y_train,
        y_test
    )

    results.append(result)

    save_model(
        result["Trained_Model"],
        "naive_bayes"
    )

    # ============================================
    # Model Comparison Report
    # ============================================

    comparison = pd.DataFrame(results)

    comparison = comparison.drop(columns=["Trained_Model"])

    comparison = comparison.sort_values(
        by="Accuracy",
        ascending=False
    )

    report_path = REPORT_DIR / "model_comparison.csv"

    comparison.to_csv(
        report_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)

    print(comparison)

    print("\nComparison report saved successfully.")
    print(report_path)
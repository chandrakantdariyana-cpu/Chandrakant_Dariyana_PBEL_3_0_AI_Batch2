"""
===========================================================
Project : AI-Based Cyber Threat Detection Framework
File    : model_evaluation.py

Author  : Chandrakant

Purpose :
Evaluate the best trained Machine Learning model.
===========================================================
"""

import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from logger import logger

# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# ==========================================================
# LOAD FILES
# ==========================================================

def main():
    logger.info("Model evaluation started.")

    print("=" * 60)
    print("Loading Saved Files...")
    print("=" * 60)

    logger.info("Loading best model.")
    best_model = joblib.load(
        MODEL_DIR / "best_model.pkl"
    )

    logger.info("Loading label encoder.")
    label_encoder = joblib.load(
        MODEL_DIR / "label_encoder.pkl"
    )

    logger.info("Loading scaler.")
    scaler = joblib.load(
        MODEL_DIR / "scaler.pkl"
    )

    logger.info("Loading X_test.")
    X_test = joblib.load(
        MODEL_DIR / "X_test.pkl"
    )

    logger.info("Loading y_test.")
    y_test = joblib.load(
        MODEL_DIR / "y_test.pkl"
    )

    logger.info("All required files loaded successfully.")
    logger.debug(f"X_test shape: {X_test.shape}")
    logger.debug(f"y_test shape: {y_test.shape}")

    print("Best Model Loaded Successfully.")
    print("Label Encoder Loaded Successfully.")
    print("Scaler Loaded Successfully.")
    print("Test Dataset Loaded Successfully.")

    # ==========================================================
    # MODEL PREDICTION
    # ==========================================================

    print("\nMaking Predictions...")

    logger.info("Making predictions using best model.")

    predictions = best_model.predict(X_test)

    logger.info("Prediction completed.")
    logger.debug(f"Total predictions: {len(predictions)}")

    # ==========================================================
    # MODEL METRICS
    # ==========================================================

    logger.info("Calculating evaluation metrics.")

    accuracy = accuracy_score(
        y_test,
        predictions
    )

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

    logger.info(
        f"Metrics -> Accuracy={accuracy:.4f}, "
        f"Precision={precision:.4f}, "
        f"Recall={recall:.4f}, "
        f"F1={f1:.4f}"
    )

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # ==========================================================
    # CLASSIFICATION REPORT
    # ==========================================================

    print("\nGenerating Classification Report...")

    logger.info("Generating classification report.")

    labels = list(range(len(label_encoder.classes_)))

    labels = list(range(len(label_encoder.classes_)))

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=label_encoder.classes_,
        zero_division=0
    )

    logger.debug(f"\n{report}")

    print(report)

    report_path = (
            REPORT_DIR /
            "classification_report.txt"
    )

    with open(
            report_path,
            "w",
            encoding="utf-8"
    ) as file:

        file.write(report)

    logger.info(f"Classification report saved at: {report_path}")

    print("\nClassification Report Saved Successfully.")
    print(report_path)

    # ==========================================================
    # CONFUSION MATRIX
    # ==========================================================

    logger.info("Generating confusion matrix.")

    labels = list(range(len(label_encoder.classes_)))

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    logger.debug(f"Confusion matrix shape: {cm.shape}")

    # ==========================================================
    # PLOT CONFUSION MATRIX
    # ==========================================================

    print("\nGenerating Confusion Matrix...")

    labels = list(range(len(label_encoder.classes_)))

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    cm_percentage = cm.astype(float)

    logger.info("Normalizing confusion matrix.")

    for i in range(len(cm_percentage)):
        row_sum = cm_percentage[i].sum()
        if row_sum > 0:
            cm_percentage[i] = (cm_percentage[i] / row_sum) * 100

    plt.figure(figsize=(16, 12))

    logger.info("Creating confusion matrix heatmap.")

    sns.heatmap(
        cm_percentage,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        linewidths=0.5,
        cbar_kws={
            "label": "Percentage (%)"
        }
    )

    plt.title(
        "Confusion Matrix (%)",
        fontsize=18
    )

    plt.xlabel(
        "Predicted Label",
        fontsize=12
    )

    plt.ylabel(
        "Actual Label",
        fontsize=12
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.yticks(
        rotation=0
    )

    plt.tight_layout()

    cm_path = RESULT_DIR / "confusion_matrix.png"

    plt.savefig(
        cm_path,
        dpi=300,
        bbox_inches="tight"
    )

    logger.info(f"Confusion matrix saved at: {cm_path}")

    plt.close()

    logger.info("Confusion matrix figure closed.")

    print("\nConfusion Matrix Saved Successfully.")
    print(cm_path)

    print("\nConfusion Matrix Saved Successfully.")
    print(cm_path)

    # ==========================================================
    # EVALUATION SUMMARY
    # ==========================================================

    logger.info("Model evaluation completed successfully.")

    print("\n" + "=" * 60)
    print("MODEL EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nGenerated Files")
    print("-" * 60)
    print(report_path)
    print(cm_path)

    print("\nBest Model Evaluated Successfully.")

if __name__ == '__main__':
    main()
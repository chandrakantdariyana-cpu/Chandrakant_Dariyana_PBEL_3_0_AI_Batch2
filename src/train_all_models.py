"""
===========================================================
Project : AI-Based Cyber Threat Detection Framework
File    : train_all_models.py

Author  : Chandrakant

Purpose :
Train, evaluate and compare Machine Learning models.
===========================================================
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from data_preprocessing import prepare_dataset
from logger import logger


# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)


# ==========================================================
# EVALUATE MODEL
# ==========================================================

def evaluate_model(
    model,
    model_name,
    X_train,
    X_test,
    y_train,
    y_test
):

    logger.info(f"Training started for {model_name}.")

    print("\n" + "=" * 60)
    print(f"Training {model_name}...")
    print("=" * 60)

    model.fit(X_train, y_train)

    logger.info(f"{model_name} trained successfully.")

    predictions = model.predict(X_test)

    logger.info(f"Prediction completed for {model_name}.")

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    logger.info(f"Classification report generated for {model_name}.")
    logger.debug(f"\n{report}")

    print(report)

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

    logger.info(
        f"{model_name} Metrics -> "
        f"Accuracy={accuracy:.4f}, "
        f"Precision={precision:.4f}, "
        f"Recall={recall:.4f}, "
        f"F1={f1:.4f}"
    )

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Model_Object": model
    }


# ==========================================================
# SAVE MODEL
# ==========================================================

def save_model(model, model_name):

    model_path = MODEL_DIR / f"{model_name}.pkl"

    logger.info(f"Saving model: {model_name}")

    joblib.dump(model, model_path)

    logger.info(f"Model saved successfully: {model_path}")

    print(f"{model_name} saved successfully.")
    print(model_path)


# ==========================================================
# MAIN
# ==========================================================

def main():
    logger.info("train_all_models.py execution started.")

    try:

        print("=" * 60)
        print("Loading Dataset...")
        print("=" * 60)

        logger.info("Preparing dataset for training.")

        (
            X_train,
            X_test,
            y_train,
            y_test,
            encoder,
            scaler
        ) = prepare_dataset()

        logger.info("Dataset prepared successfully.")
        logger.debug(f"Training shape: {X_train.shape}")
        logger.debug(f"Testing shape: {X_test.shape}")

        results = []

        models = [

            (
                "Logistic Regression",
                "logistic_regression",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            ),

            (
                "Random Forest",
                "random_forest",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=30,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )
            ),

            (
                "Decision Tree",
                "decision_tree",
                DecisionTreeClassifier(
                    random_state=42
                )
            ),

            (
                "KNN",
                "knn",
                KNeighborsClassifier(
                    n_neighbors=5
                )
            ),

            (
                "Naive Bayes",
                "naive_bayes",
                GaussianNB()
            )

        ]

        logger.info(f"Total models to train: {len(models)}")

        for display_name, file_name, model in models:
            logger.info(f"Starting model: {display_name}")

            result = evaluate_model(
                model,
                display_name,
                X_train,
                X_test,
                y_train,
                y_test
            )

            results.append(result)

            save_model(
                result["Model_Object"],
                file_name
            )

            logger.info(f"{display_name} completed successfully.")

        joblib.dump(
            X_test,
            MODEL_DIR / "X_test.pkl"
        )

        joblib.dump(
            y_test,
            MODEL_DIR / "y_test.pkl"
        )

        logger.info("X_test.pkl saved successfully.")
        logger.info("y_test.pkl saved successfully.")

        print("\nX_test.pkl saved successfully.")
        print("y_test.pkl saved successfully.")

        comparison = pd.DataFrame(results)

        comparison = comparison.drop(
            columns=["Model_Object"]
        )

        comparison = comparison.sort_values(
            by="Accuracy",
            ascending=False
        )

        comparison_path = (
                REPORT_DIR /
                "model_comparison.csv"
        )

        comparison.to_csv(
            comparison_path,
            index=False
        )

        logger.info("Model comparison report generated.")
        logger.info(f"Comparison report saved at: {comparison_path}")

        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        print(comparison)

        print("\nReport Saved Successfully.")
        print(comparison_path)

        best_result = max(
            results,
            key=lambda x: x["Accuracy"]
        )

        best_model = best_result["Model_Object"]

        logger.info(
            f"Best model selected: {best_result['Model']} "
            f"with accuracy {best_result['Accuracy']:.4f}"
        )

        joblib.dump(
            best_model,
            MODEL_DIR / "best_model.pkl"
        )

        logger.info(f"Best model saved at: {MODEL_DIR / 'best_model.pkl'}")

        print("\n" + "=" * 60)
        print("BEST MODEL")
        print("=" * 60)

        print(f"Model    : {best_result['Model']}")
        print(f"Accuracy : {best_result['Accuracy']:.4f}")

        print("\nBest model saved successfully.")
        print(MODEL_DIR / "best_model.pkl")

        print("\n" + "=" * 60)
        print("ALL MODELS TRAINED SUCCESSFULLY")
        print("=" * 60)

        logger.info("All models trained successfully.")

    except Exception as error:

        logger.exception(f"Unexpected error during model training: {error}")
        raise

if __name__ == "__main__":
    main()
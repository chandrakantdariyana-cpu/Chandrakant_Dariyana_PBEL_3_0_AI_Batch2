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


from data_preprocessing import prepare_dataset



# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent


MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)



# ==========================================================
# Model Evaluation Function
# ==========================================================


def evaluate_model(
        model,
        model_name,
        X_train,
        X_test,
        y_train,
        y_test):


    print("\n" + "=" * 60)

    print(f"Training {model_name}...")

    print("=" * 60)



    # Training

    model.fit(
        X_train,
        y_train
    )


    # Prediction

    predictions = model.predict(
        X_test
    )


    # Metrics

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

        "Trained_Model": model

    }





# ==========================================================
# Save Model
# ==========================================================


def save_model(
        model,
        model_name):


    model_path = MODEL_DIR / f"{model_name}.pkl"


    joblib.dump(
        model,
        model_path
    )


    print(
        f"{model_name} saved successfully."
    )


    print(
        model_path
    )





# ==========================================================
# Save Label Encoder
# ==========================================================


def save_label_encoder(
        encoder):


    encoder_path = MODEL_DIR / "label_encoder.pkl"


    joblib.dump(
        encoder,
        encoder_path
    )


    print("\nLabel Encoder saved successfully.")

    print(
        encoder_path
    )





# ==========================================================
# Main
# ==========================================================


if __name__ == "__main__":


    print("=" * 60)

    print("Loading Dataset...")

    print("=" * 60)



    # Load Preprocessed Dataset

    (
        X_train,
        X_test,
        y_train,
        y_test,
        encoder,
        scaler

    ) = prepare_dataset()



    results = []



    # ======================================================
    # Logistic Regression
    # ======================================================


    logistic_model = LogisticRegression(

        max_iter=1000,

        random_state=42

    )


    result = evaluate_model(

        logistic_model,

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





    # ======================================================
    # Random Forest
    # ======================================================


    random_forest_model = RandomForestClassifier(

        n_estimators=100,

        random_state=42,

        n_jobs=-1

    )


    result = evaluate_model(

        random_forest_model,

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





    # ======================================================
    # Decision Tree
    # ======================================================


    decision_tree_model = DecisionTreeClassifier(

        random_state=42

    )


    result = evaluate_model(

        decision_tree_model,

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





    # ======================================================
    # KNN
    # ======================================================


    knn_model = KNeighborsClassifier(

        n_neighbors=5

    )


    result = evaluate_model(

        knn_model,

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





    # ======================================================
    # Naive Bayes
    # ======================================================


    naive_model = GaussianNB()



    result = evaluate_model(

        naive_model,

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





    # ======================================================
    # Model Comparison Report
    # ======================================================


    comparison = pd.DataFrame(results)



    comparison = comparison.drop(

        columns=["Trained_Model"]

    )


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

    print("MODEL COMPARISON")

    print("=" * 60)



    print(comparison)



    print("\nReport Saved Successfully.")

    print(report_path)



    # Save Encoder

    save_label_encoder(
        encoder
    )



    print("\nAll Models Training Completed Successfully.")
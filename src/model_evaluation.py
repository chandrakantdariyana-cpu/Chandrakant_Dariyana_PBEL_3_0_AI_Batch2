import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)




# ============================================================
# PATH CONFIGURATION
# ============================================================

DATASET_PATH = "../data/processed/CICIDS2017_FeatureSelected.csv"

MODEL_PATH = "../models/random_forest.pkl"

OUTPUT_CM = "../results/random_forest_confusion_matrix.png"

OUTPUT_REPORT = "../results/classification_report.png"




# ============================================================
# LOAD DATASET
# ============================================================

print("="*60)
print("Loading Dataset...")
print("="*60)


df = pd.read_csv(DATASET_PATH)


print("Initial Shape :", df.shape)



# ============================================================
# CLEAN DATA
# ============================================================

print("\nCleaning Dataset...")


df.replace([np.inf,-np.inf],np.nan,inplace=True)

df.dropna(inplace=True)



print("After Cleaning :",df.shape)



# ============================================================
# SPLIT FEATURES AND LABEL
# ============================================================


X = df.drop("Label",axis=1)

y = df["Label"]



print("\nFeatures :",X.shape[1])
print("Samples :",X.shape[0])



# ============================================================
# ENCODE LABELS
# ============================================================


print("\nEncoding Labels...")


encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)



classes = encoder.classes_



# ============================================================
# SCALE FEATURES
# ============================================================


print("\nScaling Features...")


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)



print("\nDataset Ready For Evaluation.")





# ============================================================
# LOAD MODEL
# ============================================================


print("="*60)
print("Loading Random Forest Model...")
print("="*60)



model = joblib.load(MODEL_PATH)



# ============================================================
# PREDICTION
# ============================================================


print("\nMaking Predictions...")


prediction = model.predict(X_scaled)



# ============================================================
# ACCURACY
# ============================================================


accuracy = accuracy_score(
    y_encoded,
    prediction
)


print("\nAccuracy:")
print(round(accuracy,5))



# ============================================================
# CLASSIFICATION REPORT
# ============================================================


report = classification_report(
    y_encoded,
    prediction,
    target_names=classes,
    zero_division=0
)


print("\nClassification Report")
print(report)



with open(
    OUTPUT_REPORT,
    "w",
    encoding="utf-8"
) as file:
    file.write(report)



# ============================================================
# CONFUSION MATRIX
# ============================================================


cm = confusion_matrix(
    y_encoded,
    prediction
)



# Normalize safely

cm_percentage = np.zeros_like(cm,dtype=float)


for i in range(len(cm)):

    row_sum = cm[i].sum()

    if row_sum !=0:
        cm_percentage[i] = cm[i] / row_sum * 100



# ============================================================
# SHORT LABELS
# ============================================================


short_labels=[]


for label in classes:

    label = (
        label
        .replace("Web Attack � ","Web ")
        .replace("DoS ","")
        .replace("-"," ")
    )

    short_labels.append(label)



# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================


plt.figure(
    figsize=(16,12)
)



sns.heatmap(
    cm_percentage,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    xticklabels=short_labels,
    yticklabels=short_labels,
    linewidths=0.5,
    cbar_kws={
        "label":"Percentage (%)"
    }
)



plt.title(
    "Random Forest Confusion Matrix (%)",
    fontsize=18,
    pad=20
)


plt.xlabel(
    "Predicted Label",
    fontsize=14
)


plt.ylabel(
    "Actual Label",
    fontsize=14
)



plt.xticks(
    rotation=45,
    ha="right",
    fontsize=10
)


plt.yticks(
    rotation=0,
    fontsize=10
)



plt.tight_layout()



plt.savefig(
    OUTPUT_CM,
    dpi=300,
    bbox_inches="tight"
)



plt.show()



print("\nConfusion Matrix Saved Successfully.")

print("="*60)
print("Evaluation Completed")
print("="*60)
"""
===========================================================
Project : AI-Based Cyber Threat Detection Framework
File    : feature_selection.py
Author  : Chandrakant
Purpose : Remove low-variance (constant) features from the
          cleaned CICIDS2017 dataset and save the selected
          feature dataset for machine learning.
===========================================================
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_selection import VarianceThreshold


# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "CICIDS2017_Clean.csv"

OUTPUT_PATH = (
    BASE_DIR / "data" / "processed" / "CICIDS2017_FeatureSelected.csv"
)


# ==========================================================
# Feature Selection Function
# ==========================================================

def feature_selection():
    """
    Perform feature selection using Variance Threshold.
    """

    try:

        print("=" * 60)
        print("Loading cleaned dataset...")

        df = pd.read_csv(DATA_PATH)

        print("Dataset loaded successfully.")
        print(f"Dataset Shape : {df.shape}")

        # Remove extra spaces from column names
        df.columns = df.columns.str.strip()

        # Check Label column
        if "Label" not in df.columns:
            raise ValueError("'Label' column not found in dataset.")

        # Split Features and Target
        X = df.drop(columns=["Label"])
        y = df["Label"]

        print(f"\nOriginal Features : {X.shape[1]}")

        # --------------------------------------------------
        # Variance Threshold
        # --------------------------------------------------

        selector = VarianceThreshold(threshold=0)

        X_selected = selector.fit_transform(X)

        selected_columns = X.columns[selector.get_support()]

        removed_features = X.shape[1] - len(selected_columns)

        print(f"Removed Features : {removed_features}")
        print(f"Remaining Features : {len(selected_columns)}")

        # --------------------------------------------------
        # Create Final Dataset
        # --------------------------------------------------

        selected_df = pd.DataFrame(
            X_selected,
            columns=selected_columns
        )

        selected_df["Label"] = y.values

        # --------------------------------------------------
        # Save Dataset
        # --------------------------------------------------

        selected_df.to_csv(OUTPUT_PATH, index=False)

        print("\nFeature selection completed successfully.")
        print(f"Final Dataset Shape : {selected_df.shape}")
        print(f"Dataset saved to :\n{OUTPUT_PATH}")

        print("=" * 60)

    except FileNotFoundError:

        print("ERROR : Clean dataset not found.")
        print(f"Expected Location :\n{DATA_PATH}")

    except Exception as error:

        print(f"Unexpected Error : {error}")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    feature_selection()
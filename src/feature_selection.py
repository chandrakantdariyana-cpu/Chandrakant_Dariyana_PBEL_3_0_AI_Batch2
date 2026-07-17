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
import joblib
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from logger import logger


# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "CICIDS2017_Clean.csv"

OUTPUT_PATH = (
    BASE_DIR / "data" / "processed" / "CICIDS2017_FeatureSelected.csv"
)
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Feature Selection Function
# ==========================================================

def feature_selection():
    """
    Perform feature selection using Variance Threshold.
    """

    logger.info("Feature selection process started.")

    try:

        print("=" * 60)
        print("Loading cleaned dataset...")

        logger.info(f"Loading cleaned dataset from: {DATA_PATH}")

        df = pd.read_csv(DATA_PATH)

        logger.info("Dataset loaded successfully.")
        logger.debug(f"Dataset shape: {df.shape}")

        print("Dataset loaded successfully.")
        print(f"Dataset Shape : {df.shape}")

        df.columns = df.columns.str.strip()

        logger.info("Column names stripped successfully.")

        if "Label" not in df.columns:
            logger.error("'Label' column not found in dataset.")
            raise ValueError("'Label' column not found in dataset.")

        logger.info("'Label' column found successfully.")

        X = df.drop(columns=["Label"])
        y = df["Label"]

        logger.info("Features and target variable separated.")
        logger.debug(f"Feature count before selection: {X.shape[1]}")

        print(f"\nOriginal Features : {X.shape[1]}")

        selector = VarianceThreshold(threshold=0)

        logger.info("VarianceThreshold initialized.")

        X_selected = selector.fit_transform(X)

        logger.info("VarianceThreshold applied successfully.")

        selected_columns = X.columns[selector.get_support()]

        removed_features = X.shape[1] - len(selected_columns)

        logger.info(f"Removed features: {removed_features}")
        logger.info(f"Remaining features: {len(selected_columns)}")
        logger.debug(f"Selected columns: {list(selected_columns)}")
        logger.debug(f"Removed columns: {list(X.drop(columns=selected_columns).columns)}")

        print(f"Removed Features : {removed_features}")
        print(f"Remaining Features : {len(selected_columns)}")
        print(f"Removed Features: {X.drop(columns=selected_columns).columns }")

        selected_df = pd.DataFrame(
            X_selected,
            columns=selected_columns
        )

        logger.info("Selected feature dataframe created.")

        selected_df["Label"] = y.values

        joblib.dump(
            selected_columns.tolist(),
            MODEL_DIR / "feature_columns.pkl"
        )

        logger.info("Target column added back to selected dataset.")
        logger.debug(f"Final dataset shape: {selected_df.shape}")

        selected_df.to_csv(OUTPUT_PATH, index=False)

        logger.info(f"Feature selected dataset saved successfully at: {OUTPUT_PATH}")

        print("\nFeature selection completed successfully.")
        print(f"Final Dataset Shape : {selected_df.shape}")
        print(f"Dataset saved to :\n{OUTPUT_PATH}")

        print("=" * 60)

        logger.info("Feature selection process completed successfully.")

    except FileNotFoundError:

        logger.error(f"Clean dataset not found at: {DATA_PATH}")

        print("ERROR : Clean dataset not found.")
        print(f"Expected Location :\n{DATA_PATH}")

    except Exception as error:

        logger.exception(f"Unexpected error occurred: {error}")

        print(f"Unexpected Error : {error}")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    logger.info("Executing feature_selection.py")
    feature_selection()
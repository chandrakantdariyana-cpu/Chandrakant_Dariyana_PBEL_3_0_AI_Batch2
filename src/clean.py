"""
===========================================================
Project : AI-Based Cyber Threat Detection Framework
File    : clean.py
Author  : Chandrakant
Purpose : Clean the merged CICIDS2017 dataset by
          removing duplicates, missing values,
          and infinite values.
===========================================================
"""

from pathlib import Path
import pandas as pd
import numpy as np


# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "CICIDS2017_Final.csv"

OUTPUT_PATH = BASE_DIR / "data" / "processed" / "CICIDS2017_Clean.csv"


# ==========================================================
# Dataset Cleaning Function
# ==========================================================

def clean_dataset():
    """
    Load the merged dataset, clean it,
    and save the cleaned dataset.
    """

    try:

        print("=" * 60)
        print("Loading dataset...")

        df = pd.read_csv(DATA_PATH)

        print("Dataset loaded successfully.")
        print(f"Original Shape : {df.shape}")

        # --------------------------------------------------
        # Remove Duplicate Rows
        # --------------------------------------------------

        duplicate_rows = df.duplicated().sum()

        df = df.drop_duplicates()

        print(f"Duplicate Rows Removed : {duplicate_rows}")

        # --------------------------------------------------
        # Replace Infinite Values
        # --------------------------------------------------

        numeric_df = df.select_dtypes(include=[np.number])

        inf_count = np.isinf(numeric_df).sum().sum()

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        print(f"Infinite Values Found : {inf_count}")

        # --------------------------------------------------
        # Remove Missing Values
        # --------------------------------------------------

        missing_values = df.isnull().sum().sum()

        df.dropna(inplace=True)

        print(f"Missing Values Removed : {missing_values}")

        # --------------------------------------------------
        # Final Dataset Information
        # --------------------------------------------------

        print(f"Final Shape : {df.shape}")

        # --------------------------------------------------
        # Save Clean Dataset
        # --------------------------------------------------

        df.to_csv(OUTPUT_PATH, index=False)

        print()
        print("Dataset cleaned successfully.")
        print(f"Saved to : {OUTPUT_PATH}")

        print("=" * 60)

    except FileNotFoundError:

        print("ERROR : Dataset file not found.")
        print(f"Expected Location : {DATA_PATH}")

    except Exception as error:

        print(f"Unexpected Error : {error}")


# ==========================================================
# Main Function
# ==========================================================

if __name__ == "__main__":
    clean_dataset()
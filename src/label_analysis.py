"""
===========================================================
Project : AI-Based Cyber Threat Detection Framework
File    : label_analysis.py
Author  : Chandrakant
Purpose : Analyze attack labels in the cleaned dataset.
===========================================================
"""

from pathlib import Path
import pandas as pd


# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "CICIDS2017_Clean.csv"


# ==========================================================
# Label Analysis Function
# ==========================================================

def analyze_labels():
    """
    Display dataset columns and attack label distribution.
    """

    try:

        print("=" * 60)
        print("Loading cleaned dataset...")

        df = pd.read_csv(DATA_PATH)

        print("Dataset loaded successfully.")
        print(f"Dataset Shape : {df.shape}")

        # Remove extra spaces from column names
        df.columns = df.columns.str.strip()

        print("\n" + "=" * 60)
        print("Dataset Columns")
        print("=" * 60)

        for column in df.columns:
            print(column)

        print("\n" + "=" * 60)
        print("Attack Label Distribution")
        print("=" * 60)

        label_counts = df["Label"].value_counts()

        print(label_counts)

        print("\nTotal Attack Classes :", len(label_counts))

        print("=" * 60)

    except FileNotFoundError:

        print("ERROR : Clean dataset not found.")
        print(f"Expected Location : {DATA_PATH}")

    except Exception as error:

        print(f"Unexpected Error : {error}")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    analyze_labels()
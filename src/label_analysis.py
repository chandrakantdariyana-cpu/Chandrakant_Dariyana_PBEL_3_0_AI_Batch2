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
from logger import logger


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

    logger.info("Label analysis process started.")

    try:

        print("=" * 60)
        print("Loading cleaned dataset...")

        logger.info(f"Loading cleaned dataset from: {DATA_PATH}")

        df = pd.read_csv(DATA_PATH)

        logger.info("Cleaned dataset loaded successfully.")
        logger.debug(f"Dataset shape: {df.shape}")

        print("Dataset loaded successfully.")
        print(f"Dataset Shape : {df.shape}")

        df.columns = df.columns.str.strip()

        logger.info("Column names stripped successfully.")

        print("\n" + "=" * 60)
        print("Dataset Columns")
        print("=" * 60)

        logger.info("Displaying dataset columns.")

        for column in df.columns:
            logger.debug(f"Column: {column}")
            print(column)

        print("\n" + "=" * 60)
        print("Attack Label Distribution")
        print("=" * 60)

        logger.info("Calculating attack label distribution.")

        label_counts = df["Label"].value_counts()

        logger.info(f"Total attack classes found: {len(label_counts)}")

        for label, count in label_counts.items():
            logger.debug(f"{label}: {count}")

        print(label_counts)

        print("\nTotal Attack Classes :", len(label_counts))

        print("=" * 60)

        logger.info("Label analysis completed successfully.")

    except FileNotFoundError:

        logger.error(f"Clean dataset not found at: {DATA_PATH}")

        print("ERROR : Clean dataset not found.")
        print(f"Expected Location : {DATA_PATH}")

    except Exception as error:

        logger.exception(f"Unexpected error occurred: {error}")

        print(f"Unexpected Error : {error}")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    logger.info("Executing label_analysis.py")
    analyze_labels()
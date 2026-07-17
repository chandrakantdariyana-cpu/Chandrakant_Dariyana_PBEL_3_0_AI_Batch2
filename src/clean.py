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
from logger import logger


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

    logger.info("Dataset cleaning process started.")

    try:

        print("=" * 60)
        print("Loading dataset...")

        logger.info(f"Loading dataset from: {DATA_PATH}")

        df = pd.read_csv(DATA_PATH)

        logger.info("Dataset loaded successfully.")
        logger.debug(f"Original dataset shape: {df.shape}")

        print("Dataset loaded successfully.")
        print(f"Original Shape : {df.shape}")

        duplicate_rows = df.duplicated().sum()

        logger.info(f"Duplicate rows found: {duplicate_rows}")

        df = df.drop_duplicates()

        logger.debug(f"Shape after removing duplicates: {df.shape}")

        print(f"Duplicate Rows Removed : {duplicate_rows}")

        numeric_df = df.select_dtypes(include=[np.number])

        logger.info("Numeric columns selected for infinite value checking.")

        inf_count = np.isinf(numeric_df).sum().sum()

        logger.info(f"Infinite values found: {inf_count}")

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        logger.info("Infinite values replaced with NaN.")

        print(f"Infinite Values Found : {inf_count}")

        missing_values = df.isnull().sum().sum()

        logger.info(f"Missing values found: {missing_values}")

        df.dropna(inplace=True)

        logger.debug(f"Shape after removing missing values: {df.shape}")

        print(f"Missing Values Removed : {missing_values}")

        print(f"Final Shape : {df.shape}")

        logger.info(f"Final cleaned dataset shape: {df.shape}")

        df.to_csv(OUTPUT_PATH, index=False)

        logger.info(f"Clean dataset saved successfully at: {OUTPUT_PATH}")

        print()
        print("Dataset cleaned successfully.")
        print(f"Saved to : {OUTPUT_PATH}")

        print("=" * 60)

        logger.info("Dataset cleaning process completed successfully.")

    except FileNotFoundError:

        logger.error(f"Dataset file not found at: {DATA_PATH}")

        print("ERROR : Dataset file not found.")
        print(f"Expected Location : {DATA_PATH}")

    except Exception as error:

        logger.exception(f"Unexpected error occurred: {error}")

        print(f"Unexpected Error : {error}")


# ==========================================================
# Main Function
# ==========================================================

if __name__ == "__main__":
    logger.info("Executing clean.py")
    clean_dataset()
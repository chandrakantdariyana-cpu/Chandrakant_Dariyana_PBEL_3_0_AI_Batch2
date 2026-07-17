"""
===========================================================
Project : AI-Based Cyber Threat Detection Framework
File    : merge_dataset.py
Author  : Chandrakant
Purpose : Merge all CICIDS2017 CSV files into one dataset
===========================================================
"""

from pathlib import Path
import pandas as pd
from logger import logger


# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"

OUTPUT_PATH = BASE_DIR / "data" / "processed" / "CICIDS2017_Final.csv"


# ==========================================================
# Dataset Files
# ==========================================================

FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv"
]


# ==========================================================
# Merge Function
# ==========================================================

def merge_datasets():

    logger.info("Merge dataset process started.")

    try:

        print("=" * 60)
        print("Merging CICIDS2017 datasets...\n")

        logger.info(f"Raw data directory: {RAW_DATA_DIR}")
        logger.info(f"Output file: {OUTPUT_PATH}")

        dataframes = []

        for file_name in FILES:

            file_path = RAW_DATA_DIR / file_name

            logger.info(f"Loading dataset: {file_name}")
            logger.debug(f"File path: {file_path}")

            print(f"Loading : {file_name}")

            df = pd.read_csv(file_path)

            logger.info(f"{file_name} loaded successfully.")
            logger.debug(f"{file_name} shape: {df.shape}")

            print(f"Shape : {df.shape}")

            dataframes.append(df)

            logger.info(f"{file_name} added to dataframe list.")

        print("\nCombining all datasets...")

        logger.info("Combining all datasets.")

        combined_df = pd.concat(dataframes, ignore_index=True)

        logger.info("Datasets combined successfully.")
        logger.debug(f"Combined dataset shape: {combined_df.shape}")

        combined_df.to_csv(OUTPUT_PATH, index=False)

        logger.info("Merged dataset saved successfully.")
        logger.debug(f"Saved file location: {OUTPUT_PATH}")

        print("\nDataset merged successfully.")
        print(f"Final Shape : {combined_df.shape}")
        print(f"Saved to : {OUTPUT_PATH}")

        print("=" * 60)

        logger.info("Merge dataset process completed successfully.")

    except FileNotFoundError as error:

        logger.error(f"File not found: {error}")

        print(f"ERROR : File not found.\n{error}")

    except Exception as error:

        logger.exception(f"Unexpected error occurred: {error}")

        print(f"Unexpected Error : {error}")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    logger.info("Executing merge_dataset.py")
    merge_datasets()
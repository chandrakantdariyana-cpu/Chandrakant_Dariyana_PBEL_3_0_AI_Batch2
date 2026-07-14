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

    try:

        print("=" * 60)
        print("Merging CICIDS2017 datasets...\n")

        dataframes = []

        for file_name in FILES:

            file_path = RAW_DATA_DIR / file_name

            print(f"Loading : {file_name}")

            df = pd.read_csv(file_path)

            print(f"Shape : {df.shape}")

            dataframes.append(df)

        print("\nCombining all datasets...")

        combined_df = pd.concat(dataframes, ignore_index=True)

        combined_df.to_csv(OUTPUT_PATH, index=False)

        print("\nDataset merged successfully.")
        print(f"Final Shape : {combined_df.shape}")
        print(f"Saved to : {OUTPUT_PATH}")

        print("=" * 60)

    except FileNotFoundError as error:

        print(f"ERROR : File not found.\n{error}")

    except Exception as error:

        print(f"Unexpected Error : {error}")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    merge_datasets()
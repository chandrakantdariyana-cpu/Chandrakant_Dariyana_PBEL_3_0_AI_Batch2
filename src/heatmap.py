"""
===========================================================
Project : AI-Based Cyber Threat Detection Framework
File    : heatmap.py
Author  : Chandrakant
Purpose : Generate and save feature correlation heatmap
===========================================================
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from logger import logger


# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "CICIDS2017_Clean.csv"

IMAGE_DIR = BASE_DIR / "images"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Heatmap Function
# ==========================================================

def generate_heatmap():
    """
    Load cleaned dataset, calculate correlation matrix,
    generate heatmap, save image and display it.
    """

    logger.info("Heatmap generation process started.")

    try:
        print("=" * 55)
        print("Loading cleaned dataset...")

        logger.info(f"Loading cleaned dataset from: {DATA_PATH}")

        df = pd.read_csv(DATA_PATH)

        logger.info("Dataset loaded successfully.")
        logger.debug(f"Dataset shape: {df.shape}")

        print(f"Dataset Loaded Successfully")
        print(f"Shape : {df.shape}")

        numeric_df = df.select_dtypes(include="number")

        logger.info("Numeric columns selected.")
        logger.debug(f"Number of numeric features: {numeric_df.shape[1]}")

        print(f"Numeric Features : {numeric_df.shape[1]}")

        corr = numeric_df.corr()

        logger.info("Correlation matrix calculated.")
        logger.debug(f"Correlation matrix shape: {corr.shape}")

        top_corr = corr.iloc[:20, :20]

        logger.info("Top 20 features selected for heatmap.")

        plt.figure(figsize=(14, 10))

        logger.debug("Figure created with size (14, 10).")

        plt.imshow(
            top_corr,
            cmap="coolwarm",
            interpolation="nearest"
        )

        logger.info("Heatmap image generated.")

        plt.colorbar(label="Correlation")

        plt.xticks(
            range(len(top_corr.columns)),
            top_corr.columns,
            rotation=90,
            fontsize=8
        )

        plt.yticks(
            range(len(top_corr.columns)),
            top_corr.columns,
            fontsize=8
        )

        plt.title(
            "Top 20 Feature Correlation Heatmap",
            fontsize=16,
            fontweight="bold"
        )

        logger.info("Chart title, axes labels, and colorbar configured.")

        plt.tight_layout()

        logger.debug("Layout adjusted.")

        save_path = IMAGE_DIR / "heatmap.png"

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        logger.info(f"Heatmap saved successfully at: {save_path}")

        print(f"Heatmap Saved Successfully")
        print(f"Location : {save_path}")

        plt.show()

        logger.info("Heatmap displayed successfully.")

        plt.close()

        logger.info("Heatmap figure closed.")

        print("Program Finished Successfully")
        print("=" * 55)

        logger.info("Heatmap generation process completed successfully.")

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
    logger.info("Executing heatmap.py")
    generate_heatmap()
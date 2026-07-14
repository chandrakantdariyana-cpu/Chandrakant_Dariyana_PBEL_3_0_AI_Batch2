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

    try:
        print("=" * 55)
        print("Loading cleaned dataset...")

        df = pd.read_csv(DATA_PATH)

        print(f"Dataset Loaded Successfully")
        print(f"Shape : {df.shape}")

        # Select only numeric columns
        numeric_df = df.select_dtypes(include="number")

        print(f"Numeric Features : {numeric_df.shape[1]}")

        # Correlation matrix
        corr = numeric_df.corr()

        # Select first 20 features
        top_corr = corr.iloc[:20, :20]

        # Create Figure
        plt.figure(figsize=(14, 10))

        plt.imshow(
            top_corr,
            cmap="coolwarm",
            interpolation="nearest"
        )

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

        plt.tight_layout()

        save_path = IMAGE_DIR / "heatmap.png"

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(f"Heatmap Saved Successfully")
        print(f"Location : {save_path}")

        plt.show()

        plt.close()

        print("Program Finished Successfully")
        print("=" * 55)

    except FileNotFoundError:
        print("ERROR : Dataset file not found.")
        print(f"Expected Location : {DATA_PATH}")

    except Exception as error:
        print(f"Unexpected Error : {error}")


# ==========================================================
# Main Function
# ==========================================================

if __name__ == "__main__":
    generate_heatmap()
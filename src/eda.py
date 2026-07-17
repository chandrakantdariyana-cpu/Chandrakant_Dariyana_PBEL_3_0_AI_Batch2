"""
===========================================================
Project : AI-Based Cyber Threat Detection Framework
File    : attack_distribution.py
Author  : Chandrakant
Purpose : Visualize attack label distribution in the
          cleaned CICIDS2017 dataset.
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
# Attack Distribution Function
# ==========================================================

def plot_attack_distribution():

    logger.info("Attack distribution visualization started.")

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

        attack_count = df["Label"].value_counts()

        logger.info("Attack label distribution calculated.")

        for label, count in attack_count.items():
            logger.debug(f"{label}: {count}")

        print("\nGenerating attack distribution graph...")

        logger.info("Generating attack distribution graph.")

        plt.style.use("dark_background")

        logger.debug("Applied dark background style.")

        plt.figure(figsize=(12, 8))

        logger.debug("Figure created with size (12, 8).")

        attack_count = attack_count.sort_values(ascending=True)

        logger.info("Attack labels sorted in ascending order.")

        bars = plt.barh(
            attack_count.index,
            attack_count.values,
            color="royalblue",
            edgecolor="white"
        )

        logger.info("Horizontal bar chart created.")

        plt.xscale("log")

        logger.debug("X-axis set to logarithmic scale.")

        plt.title(
            "CICIDS2017 Attack Distribution",
            fontsize=20,
            fontweight="bold",
            pad=15
        )

        plt.xlabel(
            "Number of Records (Log Scale)",
            fontsize=14
        )

        plt.ylabel(
            "Attack Category",
            fontsize=14
        )

        plt.grid(
            axis="x",
            linestyle="--",
            alpha=0.4
        )

        logger.info("Chart title, labels, and grid configured.")

        for bar in bars:
            width = bar.get_width()

            plt.text(
                width * 0.95,
                bar.get_y() + bar.get_height() / 2,
                f"{int(width):,}",
                va="center",
                ha="right",
                fontsize=9,
                color="white",
                fontweight="bold"
            )

        logger.info("Bar values added to chart.")

        plt.tight_layout()

        logger.debug("Layout adjusted.")

        save_path = IMAGE_DIR / "Attack_Distribution_Horizontal.png"

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        logger.info(f"Graph saved successfully at: {save_path}")

        plt.show()

        logger.info("Graph displayed successfully.")

        print("Program completed successfully.")
        print("=" * 60)

        logger.info("Attack distribution visualization completed successfully.")

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
    logger.info("Executing attack_distribution.py")
    plot_attack_distribution()
"""
This script plots the training and validation losses as a function of the epoch.
"""
import argparse
import json

import h5py
import matplotlib.pyplot as plt
import numpy as np


def main(config):
    """
    Main function to load loss data, create a plot, and save the figure.
    """
    # Derive paths from the config
    output_folder = config["paths"]["output_folder"]
    suffix = config["paths"]["suffix"]
    artifacts_path = f"{output_folder}/losses_{suffix}.hdf5"
    plot_path = f"plots/plot_losses_{suffix}.pdf"

    # Load the loss data
    with h5py.File(artifacts_path, "r") as f:
        train_loss = np.array(f["loss_training"])
        val_loss = np.array(f["loss_validation"])

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot training and validation losses."
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config_path",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = json.load(f)

    main(config)

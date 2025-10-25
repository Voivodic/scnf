"""
This script plots a corner plot comparing the generated samples with the original data.
"""

import argparse
import json

import h5py
import numpy as np
from getdist import MCSamples, plots


def main(config):
    """
    Main function to load data, create a corner plot, and save the figure.
    """
    # Derive paths from the config
    output_folder = config["paths"]["output_folder"]
    suffix = config["paths"]["suffix"]
    samples_path = f"{output_folder}/samples_{suffix}.hdf5"
    data_path = config["paths"]["theta_path"]
    plot_path = f"{config['paths']['plot_folder']}/plot_comparison_{suffix}.pdf"

    # Load the generated samples
    with h5py.File(samples_path, "r") as f:
        samples_train = np.array(f["samples_train"])
        samples_validation = np.array(f["samples_validation"])
    print(f"Generated data shape: {samples_train.shape}")

    # Get the number of test samples
    n_test = int(
        config["splits"]["r_test"] * config["data_params"]["n_samples"]
    )

    # Load the original data
    with h5py.File(data_path, "r") as f:
        theta_data = np.array(f["thetas"])
        theta_data = theta_data[-n_test:]
    print(f"Original data shape: {theta_data.shape}")

    # Create MCSamples objects
    labels = [
        r"x_{%d}" % (i) for i in range(config["data_params"]["dimensions"])
    ]
    sample_theta = MCSamples(samples=theta_data, names=labels, labels=labels)
    train = []
    validation = []
    legends = []
    for i in [0, 9]:
        train.append(
            MCSamples(
                samples=samples_train[:, i, :], names=labels, labels=labels
            )
        )
        validation.append(
            MCSamples(
                samples=samples_validation[:, i, :], names=labels, labels=labels
            )
        )
        legends.append(f"Train Layer {i}")
        legends.append(f"Validation Layer {i}")

    # Create a plotter instance
    g = plots.get_subplot_plotter()
    g.triangle_plot(
        [sample_theta] + train + validation,
        filled=True,
        legend_labels=["Original Data"] + legends,
    )

    # Save the plot
    g.export(plot_path)
    print(f"Comparison plot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot comparison of generated samples."
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

"""
This script plots the distributions of parameters sampled from a CNF model
and compares them against the true distribution from the test set.

It uses the getdist library to generate a corner plot.
"""

import h5py
import numpy as np
from getdist import plots, MCSamples
import matplotlib.pyplot as plt


def load_samples(file_path):
    """Load samples from an HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        # Squeeze to remove the time dimension if it exists
        samples = np.array(f['samples']).squeeze()
        if samples.ndim == 1:
            samples = samples[:, np.newaxis]
    return samples


def load_true_parameters(config):
    """Load the test set of true parameters from the original data file."""
    with h5py.File(config["true_params_path"], 'r') as f:
        # The training script uses the first n_train samples, so the rest are for validation/testing
        n_train = config.get("n_train", 0) # Default to 0 if not in config
        true_params = np.array(f['parameters'][n_train:])
        if true_params.ndim == 1:
            true_params = true_params[:, np.newaxis]
    return true_params


def main():
    """
    Main function to load data, create, and save the plot.
    """
    # --- Configuration ---
    config = {
        "samples_path": "Outputs/samples.hdf5",
        "true_params_path": "data/parameters.hdf5",
        "plot_output_path": "Outputs/sample_distribution.png",
        # Optional: provide names for the parameters for better plot labels
        "param_names": [f"p_{{i}}" for i in range(8)], # Example for 8 parameters
        "n_train": 800, # Must match the n_train used in train.py
    }

    # Load the generated samples and the true parameters
    generated_samples = load_samples(config["samples_path"])
    true_samples = load_true_parameters(config)

    # Check if the number of parameters matches the provided names
    n_params = generated_samples.shape[1]
    if len(config["param_names"]) != n_params:
        print(f"Warning: Number of param_names ({len(config['param_names'])}) does not match number of parameters ({n_params}). Adjusting...")
        config["param_names"] = [f"p_{{i}}" for i in range(n_params)]

    # Create MCSamples objects for getdist
    names = config["param_names"]
    labels = [name.replace('_', '_') for name in names] # Basic LaTeX formatting

    mcs_generated = MCSamples(samples=generated_samples, names=names, labels=labels, label='Generated')
    mcs_true = MCSamples(samples=true_samples, names=names, labels=labels, label='True')

    # Create a getdist plot object
    g = plots.GetDistPlots(figsize=(10, 10))
    g.settings.axes_labelsize = 12
    g.settings.legend_fontsize = 14
    g.settings.axes_fontsize = 10

    # Generate the triangle plot
    g.triangle_plot(
        [mcs_true, mcs_generated],
        filled=True,
        legend_loc='upper right',
        line_args=[
            {'lw': 2, 'color': '#006FED'}, # Blue for true distribution
            {'lw': 2, 'color': '#E03424'}  # Red for generated distribution
        ],
        contour_colors=['#006FED', '#E03424'],
    )

    # Save the plot
    plt.savefig(config["plot_output_path"], dpi=300)

    print(f"Plot saved to {config['plot_output_path']}")


if __name__ == "__main__":
    # Ensure you have the required libraries installed:
    # pip install getdist matplotlib h5py
    main()

"""
This script samples from a trained Continuous Normalizing Flow (CNF) model.

It loads the model architecture and normalization statistics from the training
artifacts file, generates samples, and saves them to an HDF5 file.
"""

import argparse
import json

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp

from scnf import cnf


def load_normalizations(artifacts_path):
    """
    Load the training configuration and normalization statistics from an HDF5 file.

    :param artifacts_path: Path to the training artifacts HDF5 file.
    :type artifacts_path: str
    :return: A tuple containing the configuration and normalization statistics.
    :rtype: tuple
    """
    with h5py.File(artifacts_path, "r") as f:
        norm_stats = {}
        norm_group = f["normalization"]
        for key in norm_group.keys():
            norm_stats[key] = jnp.array(norm_group[key][()])

    return norm_stats


def normalize(data, mean, std):
    """Normalize data using pre-computed statistics."""
    return (data - mean) / (std + 1e-8)


def denormalize(data, mean, std):
    """Denormalize data using pre-computed statistics."""
    return data * (std + 1e-8) + mean


def main(config):
    """
    Main function to load the model, generate samples, and save the results.
    """
    # Derive paths from the sampling config
    output_folder = config["paths"]["output_folder"]
    suffix = config["paths"]["suffix"]
    artifacts_path = f"{output_folder}/normalization_{suffix}.hdf5"
    model_validation_path = f"{output_folder}/model_validation_{suffix}.eqx"
    model_training_path = f"{output_folder}/model_training_{suffix}.eqx"
    output_path = f"{output_folder}/samples_{suffix}.hdf5"

    # Load training configuration and normalization stats from artifacts
    norm_stats = load_normalizations(artifacts_path)

    # Create JAX random key
    key = jax.random.PRNGKey(config["sampling_params"]["seed"])
    model_key, sample_key = jax.random.split(key)

    # Load and normalize the grid if provided
    grid = jnp.array([])
    grid_path = config["paths"]["grid_path"]
    if grid_path:
        with h5py.File(grid_path, "r") as f:
            grid_data = jnp.array(
                f["grids"][0]
            )  # Example: use the first grid for all samples
            grid = normalize(
                grid_data, norm_stats["grid_mean"], norm_stats["grid_std"]
            )

    # Load and normalize the array if provided
    array = jnp.array([])
    array_path = config["paths"]["array_path"]
    if array_path:
        with h5py.File(array_path, "r") as f:
            array_data = jnp.array(
                f["arrays"][0]
            )  # Example: use the first array for all samples
            array = normalize(
                array_data, norm_stats["array_mean"], norm_stats["array_std"]
            )

    # Instantiate the model with the loaded architecture
    model_base = cnf.cnf(
        key=model_key,
        n_neurons=[config["data_params"]["dimensions"]]
        + config["cnf_params"]["n_neurons"]
        + [config["data_params"]["dimensions"]],
        grid_size=config["cnf_params"]["grid_size"],
        kernel_size=config["cnf_params"]["kernel_size"],
        cell_size=config["cnf_params"]["cell_size"],
        conv_irreps=config["cnf_params"]["conv_irreps"],
        n_neurons_radial=config["cnf_params"]["n_neurons_radial"],
        n_neurons_lins=config["cnf_params"]["n_neurons_lins"],
        n_neurons_array=config["cnf_params"]["n_neurons_array"],
        pooling_stride=config["cnf_params"]["pooling_stride"],
        kernel_pooling_size=config["cnf_params"]["kernel_pooling_size"],
        n_channels=config["cnf_params"]["n_channels"],
        conv_space=config["cnf_params"]["conv_space"],
        t0=config["cnf_params"]["t0"],
        t1=config["cnf_params"]["t1"],
        dt0=config["cnf_params"]["dt0"],
    )

    # Open the output file
    f = h5py.File(output_path, "w")

    # Reconstruct the model from saved parts
    model_mask = cnf.get_mask(model_base)
    diff_model_template, static_model = eqx.partition(model_base, model_mask)
    diff_model = eqx.tree_deserialise_leaves(
        model_training_path, diff_model_template
    )
    model_train = eqx.combine(diff_model, static_model)
    diff_model = eqx.tree_deserialise_leaves(
        model_validation_path, diff_model_template
    )
    model_validation = eqx.combine(diff_model, static_model)

    print("Model loaded successfully. Starting sampling...")

    # Generate samples (these are in the normalized space)
    normalized_samples = model_train.sample(
        key=sample_key,
        n_samples=config["sampling_params"]["n_samples"],
        grid=grid,
        array=array,
        n_times=config["sampling_params"]["n_times"],
    )

    # Denormalize the samples to bring them back to the original data scale
    samples = denormalize(
        normalized_samples, norm_stats["theta_mean"], norm_stats["theta_std"]
    )

    print(f"Generated {samples.shape[0]} samples.")

    # Save denormalized samples and configuration to HDF5 file
    f.create_dataset("samples_train", data=samples)

    # Generate samples (these are in the normalized space)
    normalized_samples = model_validation.sample(
        key=sample_key,
        n_samples=config["sampling_params"]["n_samples"],
        grid=grid,
        array=array,
        n_times=config["sampling_params"]["n_times"],
    )

    # Denormalize the samples to bring them back to the original data scale
    samples = denormalize(
        normalized_samples, norm_stats["theta_mean"], norm_stats["theta_std"]
    )

    print(f"Generated {samples.shape[0]} samples.")

    # Save denormalized samples and configuration to HDF5 file
    f.create_dataset("samples_validation", data=samples)

    # Close the output file
    f.close()

    print(f"Samples saved to {output_path}")


if __name__ == "__main__":
    # Get the name of the config file from the command line arguments
    parser = argparse.ArgumentParser(description="Train a new model.")
    parser.add_argument(
        "-c",
        "--config",
        dest="config_path",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    # Read the configuration file
    with open(args.config_path, "r") as f:
        config = json.load(f)

    main(config)

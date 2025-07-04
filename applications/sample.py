"""
This script samples from a trained Continuous Normalizing Flow (CNF) model.

It loads the model architecture and normalization statistics from the training
artifacts file, generates samples, and saves them to an HDF5 file.
"""

import h5py
import jax
import jax.numpy as jnp
import equinox as eqx
import ast

from scnf import cnf, inference


def load_artifacts(artifacts_path):
    """
    Load the training configuration and normalization statistics from an HDF5 file.

    :param artifacts_path: Path to the training artifacts HDF5 file.
    :type artifacts_path: str
    :return: A tuple containing the configuration and normalization statistics.
    :rtype: tuple
    """
    config = {}
    normalization_stats = {}
    with h5py.File(artifacts_path, 'r') as f:
        config_group = f['config']
        for key, value in config_group.attrs.items():
            # Safely evaluate string-stored lists back to lists
            if isinstance(value, str) and value.startswith('['):
                config[key] = ast.literal_eval(value)
            else:
                config[key] = value

        norm_group = f['normalization']
        for key in norm_group.keys():
            normalization_stats[key] = jnp.array(norm_group[key])
            
    return config, normalization_stats


def normalize(data, mean, std):
    """Normalize data using pre-computed statistics."""
    return (data - mean) / (std + 1e-8)


def denormalize(data, mean, std):
    """Denormalize data using pre-computed statistics."""
    return data * (std + 1e-8) + mean


def main():
    """
    Main function to load the model, generate samples, and save the results.
    """
    # --- Sampling Configuration ---
    sampling_config = {
        "artifacts_path": "Outputs/training_artifacts_test.h5",
        "model_path": "Outputs/Model_training_test.eqx",
        "output_path": "Outputs/samples.hdf5",
        "grid_path": "data/grids.hdf5",  # Path to grid for conditional sampling, or None
        "array_path": None, # Path to array for conditional sampling, or None
        "n_samples": 1000,
        "n_times": 1,
        "seed": 42,
    }

    # Load training configuration and normalization stats
    config, norm_stats = load_artifacts(sampling_config["artifacts_path"])

    # Create JAX random key
    key = jax.random.PRNGKey(sampling_config["seed"])
    model_key, sample_key = jax.random.split(key)

    # Load and normalize conditional data if provided
    grid = jnp.array([])
    if sampling_config["grid_path"]:
        with h5py.File(sampling_config["grid_path"], 'r') as f:
            grid_data = jnp.array(f['grids'][0]) # Example: use the first grid for all samples
            grid = normalize(grid_data, norm_stats['grid_mean'], norm_stats['grid_std'])

    array = jnp.array([])
    if sampling_config["array_path"]:
        with h5py.File(sampling_config["array_path"], 'r') as f:
            array_data = jnp.array(f['arrays'][0]) # Example: use the first array for all samples
            array = normalize(array_data, norm_stats['array_mean'], norm_stats['array_std'])

    # Instantiate the model with the loaded architecture
    model = cnf.cnf(
        key=model_key,
        **{k: v for k, v in config.items() if k in cnf.cnf.__init__.__code__.co_varnames}
    )

    # Reconstruct the model from saved parts
    model_mask = inference.get_mask(model)
    diff_model_template, static_model = eqx.partition(model, model_mask)
    diff_model = eqx.tree_deserialise_leaves(sampling_config["model_path"], diff_model_template)
    model = eqx.combine(diff_model, static_model)

    print("Model loaded successfully. Starting sampling...")

    # Generate samples (these are in the normalized space)
    normalized_samples = model.sample(
        key=sample_key,
        n_samples=sampling_config["n_samples"],
        grid=grid,
        array=array,
        n_times=sampling_config["n_times"],
    )

    # Denormalize the samples to bring them back to the original data scale
    samples = denormalize(normalized_samples, norm_stats['theta_mean'], norm_stats['theta_std'])

    print(f"Generated {samples.shape[0]} samples.")

    # Save denormalized samples and configuration to HDF5 file
    with h5py.File(sampling_config["output_path"], 'w') as f:
        f.create_dataset('samples', data=samples)
        config_group = f.create_group('config')
        for key, value in sampling_config.items():
            if value is not None:
                config_group.attrs[key] = str(value) if isinstance(value, list) else value
    
    print(f"Samples saved to {sampling_config['output_path']}")


if __name__ == "__main__":
    main()

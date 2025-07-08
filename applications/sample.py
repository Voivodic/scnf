"""
This script samples from a trained Continuous Normalizing Flow (CNF) model.

It loads the model architecture and normalization statistics from the training
artifacts file, generates samples, and saves them to an HDF5 file.
"""

import h5py
import jax
import jax.numpy as jnp
import equinox as eqx
import json
import argparse

from scnf import cnf, inference


def load_model_artifacts(artifacts_path):
    """
    Load the training configuration and normalization statistics from an HDF5 file.

    :param artifacts_path: Path to the training artifacts HDF5 file.
    :type artifacts_path: str
    :return: A tuple containing the configuration and normalization statistics.
    :rtype: tuple
    """
    with h5py.File(artifacts_path, 'r') as f:
        config_json = f['config'].attrs['json']
        config = json.loads(config_json)

        norm_stats = {}
        norm_group = f['normalization']
        for key in norm_group.keys():
            norm_stats[key] = jnp.array(norm_group[key][()])
            
    return config, norm_stats


def normalize(data, mean, std):
    """Normalize data using pre-computed statistics."""
    return (data - mean) / (std + 1e-8)


def denormalize(data, mean, std):
    """Denormalize data using pre-computed statistics."""
    return data * (std + 1e-8) + mean


def main(args):
    """
    Main function to load the model, generate samples, and save the results.
    """
    # Load sampling configuration
    with open(args.config_path, 'r') as f:
        sampling_config = json.load(f)

    # Derive paths from the sampling config
    output_folder = sampling_config['paths']['output_folder']
    suffix = sampling_config['paths']['suffix']
    artifacts_path = f"{output_folder}/training_artifacts_{suffix}.hdf5"
    model_training_path = f"{output_folder}/Model_training_{suffix}.eqx"
    model_validation_path = f"{output_folder}/Model_validation_{suffix}.eqx"
    output_path = f"{output_folder}/samples_{suffix}.hdf5"

    # Load training configuration and normalization stats from artifacts
    train_config, norm_stats = load_model_artifacts(artifacts_path)

    # Create JAX random key
    key = jax.random.PRNGKey(args.seed)
    model_key, sample_key = jax.random.split(key)

    # Load and normalize conditional data if provided
    grid = jnp.array([])
    grid_path = train_config['paths']['grid_path']
    if grid_path:
        with h5py.File(grid_path, 'r') as f:
            grid_data = jnp.array(f['grids'][0])  # Example: use the first grid for all samples
            grid = normalize(grid_data, norm_stats['grid_mean'], norm_stats['grid_std'])

    array = jnp.array([])
    array_path = train_config['paths']['array_path']
    if array_path:
        with h5py.File(array_path, 'r') as f:
            array_data = jnp.array(f['arrays'][0])  # Example: use the first array for all samples
            array = normalize(array_data, norm_stats['array_mean'], norm_stats['array_std'])

    # Instantiate the model with the loaded architecture
    model = cnf.cnf(
        key=model_key,
        **train_config["cnf_params"]
    )

    # Reconstruct the model from saved parts
    model_mask = inference.get_mask(model)
    diff_model_template, static_model = eqx.partition(model, model_mask)
    diff_model = eqx.tree_deserialise_leaves(model_path, diff_model_template)
    model = eqx.combine(diff_model, static_model)

    print("Model loaded successfully. Starting sampling...")

    # Generate samples (these are in the normalized space)
    normalized_samples = model.sample(
        key=sample_key,
        n_samples=args.n_samples,
        grid=grid,
        array=array,
        n_times=args.n_times,
    )

    # Denormalize the samples to bring them back to the original data scale
    samples = denormalize(normalized_samples, norm_stats['theta_mean'], norm_stats['theta_std'])

    print(f"Generated {samples.shape[0]} samples.")

    # Save denormalized samples and configuration to HDF5 file
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('samples', data=samples)
        config_group = f.create_group('config')
        # Save sampling arguments
        sampling_args = vars(args)
        sampling_args['derived_artifacts_path'] = artifacts_path
        sampling_args['derived_model_path'] = model_path
        for key, value in sampling_args.items():
            if value is not None:
                config_group.attrs[key] = str(value) if isinstance(value, list) else value
    
    print(f"Samples saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample from a trained CNF model.')
    parser.add_argument('-c', '--config', dest='config_path', type=str, required=True, help='Path to the configuration file for sampling.')
    parser.add_argument('--n-samples', type=int, default=1000, help='Number of samples to generate.')
    parser.add_argument('--n-times', type=int, default=1, help='Number of times to sample.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling.')
    args = parser.parse_args()
    main(args)

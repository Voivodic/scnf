"""
This script samples from a trained Continuous Normalizing Flow (CNF) model.

It takes the path to a saved model, the number of samples to generate,
and the model configuration as input. The samples are saved to an HDF5 file.
"""

import h5py
import jax
import jax.numpy as jnp
import equinox as eqx

from scnf import cnf


def main():
    """
    Main function to parse arguments, load the model, sample, and save results.
    """
    # --- Configuration ---
    config = {
        # File paths
        "model_path": "/path/to/your/model.eqx",
        "output_path": "/path/to/your/samples.hdf5",
        "grid_path": None,
        "array_path": None,

        # Sampling parameters
        "n_samples": 1000,
        "n_times": 1,
        "seed": 42,

        # Model architecture parameters
        "n_neurons": [64, 64, 64],
        "grid_size": [],
        "kernel_size": 3,
        "cell_size": 1.0,
        "conv_irreps": [],
        "n_neurons_radial": [4],
        "n_neurons_lins": [],
        "n_neurons_array": [],
        "pooling_stride": 2,
        "kernel_pooling_size": 3,
        "n_channels": [],
        "conv_space": "configuration",
        "t0": 0.0,
        "t1": 1.0,
        "dt0": 0.1,
    }

    # Create JAX random key
    key = jax.random.PRNGKey(config["seed"])
    model_key, sample_key = jax.random.split(key)

    # Load conditional data if provided
    grid = jnp.load(config["grid_path"]) if config["grid_path"] else jnp.array([])
    array = jnp.load(config["array_path"]) if config["array_path"] else jnp.array([])

    # Instantiate the model with the specified architecture
    model = cnf.cnf(
        key=model_key,
        n_neurons=config["n_neurons"],
        grid_size=config["grid_size"],
        kernel_size=config["kernel_size"],
        cell_size=config["cell_size"],
        conv_irreps=config["conv_irreps"],
        n_neurons_radial=config["n_neurons_radial"],
        n_neurons_lins=config["n_neurons_lins"],
        n_neurons_array=config["n_neurons_array"],
        pooling_stride=config["pooling_stride"],
        kernel_pooling_size=config["kernel_pooling_size"],
        n_channels=config["n_channels"],
        conv_space=config["conv_space"],
        t0=config["t0"],
        t1=config["t1"],
        dt0=config["dt0"],
    )

    # Reconstruct the model from saved parts
    model_mask = cnf.get_mask(model)
    diff_model_template, static_model = eqx.partition(model, model_mask)
    
    # Load the differentiable (trained) part of the model
    diff_model = eqx.tree_deserialise_leaves(config["model_path"], diff_model_template)

    # Combine the trained and static parts
    model = eqx.combine(diff_model, static_model)

    print("Model loaded successfully. Starting sampling...")

    # Generate samples
    samples = model.sample(
        key=sample_key,
        n_samples=config["n_samples"],
        grid=grid,
        array=array,
        n_times=config["n_times"],
    )

    print(f"Generated {samples.shape[0]} samples.")

    # Save samples and configuration to HDF5 file
    with h5py.File(config["output_path"], 'w') as f:
        f.create_dataset('samples', data=samples)
        
        config_group = f.create_group('config')
        for key, value in config.items():
            if value is not None:
                # h5py attributes can't store lists, so convert to string
                if isinstance(value, list):
                    config_group.attrs[key] = str(value)
                else:
                    config_group.attrs[key] = value
    
    print(f"Samples saved to {config['output_path']}")


if __name__ == "__main__":
    main()

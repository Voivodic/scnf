"""
This script trains a Continuous Normalizing Flow (CNF) model using the
inference class from scnf.inference.

It supports unconditional training as well as training conditioned on grid and/or
array data. The configuration for the model and training is set directly in this file.
"""

import h5py
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from scnf import cnf, inference

# --- Data Loaders and Normalization ---

def get_normalization_stats(data, n_train):
    """
    Compute the mean and standard deviation of the training data.

    :param data: The full dataset.
    :type data: jnp.ndarray
    :param n_train: The number of training samples.
    :type n_train: int
    :return: A tuple containing the mean and standard deviation.
    :rtype: tuple
    """
    train_data = data[:n_train]
    mean = jnp.mean(train_data, axis=0)
    std = jnp.std(train_data, axis=0)
    return mean, std

def get_data_loaders(config):
    """
    Create data loader functions that read from HDF5 files and apply normalization.

    :param config: The configuration dictionary.
    :type config: dict
    :return: A tuple of data loaders and normalization stats.
    :rtype: tuple
    """
    # Open HDF5 files
    theta_file = h5py.File(config["theta_path"], 'r')
    grid_file = h5py.File(config["grid_path"], 'r') if config["grid_path"] else None
    array_file = h5py.File(config["array_path"], 'r') if config["array_path"] else None

    theta_dataset = theta_file['parameters']
    grid_dataset = grid_file['grids'] if grid_file else None
    array_dataset = array_file['arrays'] if array_file else None

    # Compute normalization stats from the training set
    theta_mean, theta_std = get_normalization_stats(theta_dataset, config["n_train"])
    grid_mean, grid_std = (get_normalization_stats(grid_dataset, config["n_train"]) 
                           if grid_dataset else (0, 1))
    array_mean, array_std = (get_normalization_stats(array_dataset, config["n_train"]) 
                             if array_dataset else (0, 1))

    def normalize(data, mean, std):
        return (data - mean) / (std + 1e-8) # Add epsilon to avoid division by zero

    def data_loader_theta(index):
        return normalize(jnp.array(theta_dataset[index]), theta_mean, theta_std)

    def data_loader_grid(index):
        if grid_dataset is not None:
            return normalize(jnp.array(grid_dataset[index]), grid_mean, grid_std)
        return jnp.array([])

    def data_loader_array(index):
        if array_dataset is not None:
            return normalize(jnp.array(array_dataset[index]), array_mean, array_std)
        return jnp.array([])

    normalization_stats = {
        'theta_mean': theta_mean,
        'theta_std': theta_std,
        'grid_mean': grid_mean,
        'grid_std': grid_std,
        'array_mean': array_mean,
        'array_std': array_std,
    }

    return data_loader_theta, data_loader_grid, data_loader_array, normalization_stats


def save_training_artifacts(config, normalization_stats):
    """
    Save the training configuration and normalization statistics to an HDF5 file.

    :param config: The training configuration.
    :type config: dict
    :param normalization_stats: The normalization statistics.
    :type normalization_stats: dict
    """
    output_path = f"{config['output_folder']}/training_artifacts_{config['suffix']}.h5"
    with h5py.File(output_path, 'w') as f:
        config_group = f.create_group('config')
        for key, value in config.items():
            if value is not None:
                if isinstance(value, list):
                    config_group.attrs[key] = str(value)
                else:
                    config_group.attrs[key] = value

        norm_group = f.create_group('normalization')
        for key, value in normalization_stats.items():
            norm_group.create_dataset(key, data=value)
    
    print(f"Training artifacts saved to {output_path}")


def main():
    """
    Main function to configure and run the training process.
    """
    # --- Configuration ---
    config = {
        # File paths
        "theta_path": "data/parameters.hdf5",
        "grid_path": "data/grids.hdf5",  # Path to grid data, or None
        "array_path": None,  # Path to array data, or None
        "output_folder": "Outputs",
        "suffix": "test",

        # Data parameters
        "n_train": 800,
        "n_validation": 200,

        # Training hyperparameters
        "seed": 42,
        "n_epochs": 100,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "print_every": 1,
        "lr_limit": 1e-6,
        "poly_order": 1,
        "alpha_reg": 0.01,

        # Model architecture parameters
        "n_neurons": [8, 64, 64, 8],
        "grid_size": [32, 32, 32],
        "kernel_size": 3,
        "cell_size": 1.0,
        "conv_irreps": ["1x0e", "2x1o", "1x2e"],
        "n_neurons_radial": [8, 8],
        "n_neurons_lins": [16, 16],
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
    model_key, train_key = jax.random.split(key)

    # Get data loaders and normalization stats
    data_loader_theta, data_loader_grid, data_loader_array, norm_stats = get_data_loaders(config)

    # Instantiate the model
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

    # Instantiate the inference class
    trainer = inference.inference(
        key=model_key,
        n_train=config["n_train"],
        n_validation=config["n_validation"],
        data_loader_theta=data_loader_theta,
        data_loader_grid=data_loader_grid,
        data_loader_array=data_loader_array,
        model=model,
        folder_name=config["output_folder"],
    )

    # Set up the optimizer
    optimizer = optax.adamw(learning_rate=config["learning_rate"], weight_decay=config["weight_decay"])

    print("Starting training...")

    # Run the training
    trainer.train(
        key=train_key,
        n_epochs=config["n_epochs"],
        batch_size=config["batch_size"],
        optim=optimizer,
        print_every=config["print_every"],
        suffix=config["suffix"],
        lr_limit=config["lr_limit"],
        poly_order=config["poly_order"],
        alpha_reg=config["alpha_reg"],
    )

    print("Training complete.")

    # Save training artifacts
    save_training_artifacts(config, norm_stats)


if __name__ == "__main__":
    main()

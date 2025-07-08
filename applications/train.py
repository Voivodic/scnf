"""
This script trains a Continuous Normalizing Flow (CNF) model using the
inference class from scnf.inference.

It supports unconditional training as well as training conditioned on grid and/or
array data. The configuration for the model and training is set directly in this file.
"""
# Base libraries used
import h5py
import jax
import jax.numpy as jnp
import optax
import json
import argparse

# Import the scnf library
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

def get_data_loaders(n_samples, config):
    """
    Create data loader functions that read from HDF5 files and apply normalization.

    :param config: The configuration dictionary.
    :type config: dict
    :return: A tuple of data loaders and normalization stats.
    :rtype: tuple
    """
    # Open HDF5 files
    theta_file = h5py.File(config["paths"]["theta_path"], 'r')
    grid_file = h5py.File(config["paths"]["grid_path"], 'r') if config["paths"]["grid_path"] else None
    array_file = h5py.File(config["paths"]["array_path"], 'r') if config["paths"]["array_path"] else None

    theta_dataset = theta_file['thetas']
    grid_dataset = grid_file['grids'] if grid_file else None
    array_dataset = array_file['arrays'] if array_file else None

    # Get the number of samples and check if the datasets are compatible
    if theta_dataset.shape[0] < n_samples:
        raise ValueError("The number of samples in thetas.hdf5 is less than the required number of samples.")
    if grid_dataset is not None and grid_dataset.shape[0] < n_samples:
        raise ValueError("The number of samples in grids.hdf5 is less than the required number of samples.")
    if array_dataset is not None and array_dataset.shape[0] < n_samples:
        raise ValueError("The number of samples in arrays.hdf5 is less than the required number of samples.")

    # Compute normalization stats from the training set
    n_train = int(n_samples * config["splits"]["r_train"])
    theta_mean, theta_std = get_normalization_stats(theta_dataset, n_train)
    grid_mean, grid_std = (get_normalization_stats(grid_dataset, n_train) 
                           if grid_dataset else (0, 1))
    array_mean, array_std = (get_normalization_stats(array_dataset, n_train) 
                             if array_dataset else (0, 1))

    def normalize(data, mean, std):
        return (data - mean) / (std + 1e-8) # Add epsilon to avoid division by zero

    def data_loader_theta(index):
        return normalize(jnp.array(theta_dataset)[index,:], theta_mean, theta_std)

    def data_loader_grid(index):
        if grid_dataset is not None:
            return normalize(jnp.array(grid_dataset)[index,:], grid_mean, grid_std)
        return jnp.array([])

    def data_loader_array(index):
        if array_dataset is not None:
            return normalize(jnp.array(array_dataset)[index,:], array_mean, array_std)
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
    output_path = f"{config['paths']['output_folder']}/training_artifacts_{config['paths']['suffix']}.hdf5"
    with h5py.File(output_path, 'w') as f:
        config_group = f.create_group('config')
        config_group.attrs['json'] = json.dumps(config)

        norm_group = f.create_group('normalization')
        for key, value in normalization_stats.items():
            norm_group.create_dataset(key, data=value)
    
    print(f"Training artifacts saved to {output_path}")


def main(n_samples, dim, config_path):
    """
    Main function to configure and run the training process.
    """
    # --- Configuration ---
    with open(config_path, "r") as f:
        config = json.load(f)

    # Create JAX random key
    key = jax.random.PRNGKey(config["training_params"]["seed"])
    model_key, train_key = jax.random.split(key)

    # Get data loaders and normalization stats
    data_loader_theta, data_loader_grid, data_loader_array, norm_stats = get_data_loaders(n_samples, config)

    # Instantiate the model
    model = cnf.cnf(
        key=model_key,
        n_neurons=[dim]+config["cnf_params"]["n_neurons"]+[dim],
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

    # Instantiate the inference class
    trainer = inference.inference(
        key=model_key,
        n_train=int(config["splits"]["r_train"]*n_samples),
        n_validation=int(config["splits"]["r_validation"]*n_samples),
        data_loader_theta=data_loader_theta,
        data_loader_grid=data_loader_grid,
        data_loader_array=data_loader_array,
        model=model,
        folder_name=config["paths"]["output_folder"],
    )

    # Set up the optimizer
    optimizer = optax.adamw(learning_rate=config["training_params"]["learning_rate"], weight_decay=config["training_params"]["weight_decay"])

    print("Starting training...")

    # Run the training
    trainer.train(
        key=train_key,
        n_epochs=config["training_params"]["n_epochs"],
        batch_size=config["training_params"]["batch_size"],
        optim=optimizer,
        print_every=config["training_params"]["print_every"],
        suffix=config["paths"]["suffix"],
        lr_limit=config["training_params"]["lr_limit"],
        poly_order=config["training_params"]["poly_order"],
        alpha_reg=config["training_params"]["alpha_reg"],
    )

    print("Training complete.")

    # Save training artifacts
    save_training_artifacts(config, norm_stats)


if __name__ == "__main__":
    # Get the name of the config file from the command line arguments
    parser = argparse.ArgumentParser(description='Train a new model.')
    parser.add_argument('-n', '--num-samples', dest='N', type=int, default=1_000, help='Number of samples to generate.')
    parser.add_argument('-d', '--dim', dest='D', type=int, default=3, help='Number of dimensions of the data.')
    parser.add_argument('-c', '--config', dest='config_path', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    # Run the training
    main(args.N, args.D, args.config_path)

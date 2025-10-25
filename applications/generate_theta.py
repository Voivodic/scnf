import argparse
import json

import h5py as h5
import jax
import jax.numpy as jnp


def generate_data(config, key):
    """Generates N D-dimensional random numbers, where each dimension is from a
    Beta distribution with different parameters.

    :param config: Configuration dictionary.
    :type config: dict
    :param key: JAX random key.
    :type key: jax.random.PRNGKey
    :return: An array of shape (N, D) with the generated data.
    :rtype: jnp.ndarray
    """
    # Get the parameters for generating the data
    N = config["data_params"]["n_samples"]
    D = config["data_params"]["dimensions"]
    a_params = jnp.array(config["data_params"]["beta_params"]["a"])
    b_params = jnp.array(config["data_params"]["beta_params"]["b"])

    if a_params.shape != (D,):
        raise ValueError(
            f"The length of 'a' in beta_params should be equal to the number of dimensions D={D}."
        )
    if b_params.shape != (D,):
        raise ValueError(
            f"The length of 'b' in beta_params should be equal to the number of dimensions D={D}."
        )

    # Split the key for each dimension to generate independent samples
    keys = jax.random.split(key, D)

    # Generate samples for each dimension and stack them
    beta_samples = jnp.stack(
        [
            jax.random.beta(k, a, b, shape=(N,))
            for k, a, b in zip(keys, a_params, b_params)
        ],
        axis=1,
    )

    return beta_samples


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate data for the model.")
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

    # Create a random key
    key = jax.random.PRNGKey(config["data_params"]["seed"])

    # Generate the data
    data = generate_data(config, key)

    # Print some information about the generated data
    print(f"Generated data shape: {data.shape}")

    # Save the data to a HDF5
    with h5.File(config["paths"]["theta_path"], "w") as f:
        f.create_dataset("thetas", data=data)

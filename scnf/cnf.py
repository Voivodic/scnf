"""
This module implements the continuous normalizing flow
"""

import equinox as eqx

# Import the jax related modules
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float, PRNGKeyArray, Union

# Import the modules used
from . import layers


# Define the vector field used into the neural ordinary differential equation
class vector_field(eqx.Module):
    layers: list

    # Initialize the vector field
    def __init__(
        self,
        key: PRNGKeyArray,
        x_size: Union[int, list] = 0,
        n_neurons: list = [3, 16, 16, 3],
        kernel_sizes: Union[int, list] = 3,
        stride_sizes: Union[int, list] = 1,
        cell_size: float = 1.0,
        convs_irreps: Union[list, None] = None,
        n_neurons_lins: Union[list, None] = None,
        n_neurons_radial: list = [4],
        pooling_sizes: Union[list, int] = 2,
        n_channels: Union[list, None] = None,
    ):
        """
        Initialize the class for the vector field of the ODE.

        :param key: Key for random number generation
        :type key: PRNGKeyArray
        :param x_size: Size of the conditional data
        :type in_size: Union[int, list]
        :param kernel_sizes: Size of the convolutional kernels (when applicable)
        :type kernel_sizes: Union[int, list]
        :param stride_sizes: Stride for the convolutions (when applicable)
        :type stride_sizes: Union[int, list]
        :param cell_size: Size of the cell in the input grid (when applicable)
        :type cell_size: float
        :param conv_irreps: Irreps for the convolutional layers (when applicable)
        :type conv_irreps: Union[list, None]
        :param n_neurons_lins: Number of neurons for the linear layers (when applicable)
        :type n_neurons_lins: Union[list, None]
        :param n_neurons_radial: Number of neurons for the radial part (when applicable)
        :type n_neurons_radial: list
        :param pooling_sizes: Size of the pooling layer (when applicable)
        :type pooling_sizes: Union[int, list]
        :param n_channels: Number of chennels of each layer of the convolutions (when applicable)
        :type n_channels: Union[list, None] 
        """
        # Check if the first and last layers have the same number of neurons
        if n_neurons[0] != n_neurons[-1]:
            raise ValueError(
                "The first and last layers must have the same number of neurons!"
            )

        # Get the number of layers
        n_layers = len(n_neurons) - 1

        # Set the keys
        keys = jrandom.split(key, n_layers)

        # Get the dimension of the conditional x
        if x_size is not list:
            x_dim = 0
        else:
            x_dim = len(x_size)

        # Initialize all layers that will concatanete the information
        for i in range(n_layers):
            self.layers.append(
                [
                    layers.concat_layer(
                        key=keys[i],
                        in_size=n_neurons[i],
                        out_size=n_neurons[i + 1],
                        dimension=x_dim,
                        kernel_size=kernel_sizes[i],
                        stride_size=stride_sizes[i],
                        cell_size=cell_size,
                        conv_irreps=(
                            convs_irreps[i] if convs_irreps is not None else None
                        ),
                        n_neurons_lins=(
                            n_neurons_lins[i] if n_neurons_lins is not None else None
                        ),
                        n_neurons_radial=n_neurons_radial[i],
                        pooling_size=(
                            pooling_sizes[i]
                            if isinstance(pooling_sizes, list)
                            else pooling_sizes
                        ),
                        n_channels=n_channels[i] if n_channels is not None else None,
                    )
                ]
            )

    # Compute the vector field for a given vector and time
    def __call__(self, t: float, y: Float[Array, "y_size"], args):
        """ """
        pass

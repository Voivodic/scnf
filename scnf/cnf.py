"""
This module implements the continuous normalizing flow
"""

# Import the main libraries
import equinox as eqx
import diffrax as df

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
    concatenation_layers: list
    compression_grid_layers: list
    compressed_grid_data: list
    compression_array_layers: list
    compressed_array_data: list

    # Initialize the vector field
    def __init__(
        self,
        key: PRNGKeyArray,
        n_neurons: list,
        grid_size: Union[list, int] = None,
        kernel_size: int = 3,
        cell_size: float = 1.0,
        conv_irreps: Union[list, None] = None,
        n_neurons_radial: list = [4],
        n_neurons_lins: Union[list, None] = None,
        n_neurons_array: Union[list, None] = None,
        pooling_stride: int = 2,
        kernel_pooling_size: int = 3,
        n_channels: Union[list, None] = None,
        conv_space: str = "configuration",
    ):
        """ """
        # Check the parity of the kernel
        if conv_space == "configuration" and kernel_size % 2 == 0:
            raise ValueError(
                "The kernel size must be odd for the configuration space cnn"
            )
        elif conv_space == "fourier" and kernel_size % 2 != 0:
            raise ValueError("The kernel size must be even for the fourier space cnn")

        # Get the size of the compressed grid and array
        if n_neurons_lins is not None:
            compressed_grid_size = n_neurons_lins[-1]
        else:
            compressed_grid_size = 0
        if n_neurons_array is not None:
            compressed_array_size = n_neurons_array[-1]
        else:
            compressed_array_size = 0

        # Get some parameters
        n_layers = len(n_neurons) - 1

        # Split the key
        key_concat, key_compress_grid, key_compress_array = jrandom.split(key, 3)

        # Initialize the concatenation layers
        keys_concat = jrandom.split(key_concat, n_layers)
        self.concatenation_layers = []
        for i in range(n_layers):
            self.concatenation_layers.append(
                layers.concat_layer(
                    key=keys_concat,
                    in_size=n_neurons[i],
                    out_size=n_neurons[i + 1],
                    compressed_grid_size=compressed_grid_size,
                    compressed_array_size=compressed_array_size,
                )
            )

        # Set the arrays with the compressed data and compression layers
        self.compressed_grid_data = []
        self.compressed_array_data = []
        self.compression_grid_layers = []
        self.compression_array_layers = []
        for i in range(n_layers):
            self.compressed_grid_data.append(jnp.array([]))
            self.compressed_array_data.append(jnp.array([]))
            self.compression_grid_layers.append(None)
            self.compression_array_layers.append(None)

        # Initialize the compression array layers
        if compressed_array_size > 0:
            keys_compress_array = jrandom.split(key_compress_array, n_layers)
            for i in range(n_layers):
                self.compression_array_layers[i] = layers.compress_array(
                    key=keys_compress_array[i], n_neurons_lins=n_neurons_array
                )

        # Initialize the compression grid layers
        if compressed_grid_size > 0:
            keys_compress_grid = jrandom.split(key_compress_grid, n_layers)
            for i in range(n_layers):
                if conv_irreps is None:
                    dimension = len(grid_size)
                    self.compression_grid_layers[i] = layers.compress_nd(
                        key=keys_compress_grid[i],
                        dimension=dimension,
                        kernel_size=kernel_size,
                        conv_channels=n_channels,
                        n_neurons_lins=n_neurons_lins,
                        pooling_stride=pooling_stride,
                        kernel_pooling_size=kernel_pooling_size,
                    )
                else:
                    if conv_space == "configuration":
                        self.compression_grid_layers[i] = layers.compress_3d_e3(
                            key=keys_compress_grid[i],
                            kernel_size=kernel_size,
                            cell_size=cell_size,
                            conv_irreps=conv_irreps,
                            n_neurons_lins=n_neurons_lins,
                            n_neurons_radial=n_neurons_radial,
                            pooling_stride=pooling_stride,
                            kernel_pooling_size=kernel_pooling_size,
                        )
                    elif conv_space == "fourier":
                        if grid_size is None:
                            raise ValueError(
                                "The grid_size must be provided for the fourier space convolutions!"
                            )

                        self.compression_grid_layers[i] = layers.compress_fourier_3d_e3(
                            key=keys_compress_grid[i],
                            grid_size=grid_size,
                            cell_size=cell_size,
                            conv_irreps=conv_irreps,
                            n_neurons_lins=n_neurons_lins,
                            n_neurons_radial=n_neurons_radial,
                            downsampling_factor=pooling_stride,
                        )

    # Compute the kernels of the cnns
    def compute_kernels(self):
        """ """
        for conv in self.convs:
            conv.compute_kernels()

    # Compress the array
    def compress_array(self, array: Float[Array, "array_size"]):
        """ """
        for i, compress_array in enumerate(self.compress_array_layers):
            self.compressed_array_data[i] = compress_array(array)

    # Compress the grid
    def compress_grid(
        self, grid: Float[Array, "grid_size grid_size grid_size channel_size"]
    ):
        """ """
        for i, compress_grid in enumerate(self.compression_grid_layers):
            self.compressed_grid_data[i] = compress_grid(grid)

    # Compute the vector field for a given vector and time
    def __call__(
        self, t: float, theta: Float[Array, "in_size"], args
    ) -> Float[Array, "in_size"]:
        """ """
        # Compute the concatenations
        for i, concat_layer in enumerate(self.concatenation_layers):
            theta = concat_layer(
                t, theta, self.compressed_grid_data[i], self.compressed_array_data[i]
            )
            theta = jnn.gelu(theta)

        return theta

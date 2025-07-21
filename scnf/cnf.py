"""
This module implements the continuous normalizing flow
"""

# Import standard libraries
from typing import Any, Callable, Tuple, cast

# Import jax related libraries
import diffrax as df
import equinox as eqx

# Import the jax modules
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree as jtree
from jaxtyping import Array, Float, PRNGKeyArray, Scalar

# Import the modules used
from . import layers


# Define the vector field used into the neural ordinary differential equation
class vector_field_layer(eqx.Module):
    """
    This class implements the vector field for the continuous normalizing flow,
    which is a neural ordinary differential equation (NODE).

    It processes input data (theta) by concatenating it with compressed
    grid and array data through a series of layers.
    """

    concatenation_layers: list[layers.concat_layer]
    compression_grid_layers: list[
        layers.compress_nd
        | layers.compress_3d_e3
        | layers.compress_fourier_3d_e3
        | layers.no_compression
    ]
    compression_array_layers: list[
        layers.compress_array | Callable[[jnp.ndarray], jnp.ndarray]
    ]

    # Initialize the vector field
    def __init__(
        self,
        key: PRNGKeyArray,
        n_neurons: list[int],
        grid_size: list[int] = [],
        kernel_size: int = 3,
        cell_size: float = 1.0,
        conv_irreps: list[str] = [],
        n_neurons_radial: list[int] = [4],
        n_neurons_lins: list[int] = [],
        n_neurons_array: list[int] = [],
        pooling_stride: int = 2,
        kernel_pooling_size: int = 3,
        n_channels: list[int] = [],
        conv_space: str = "configuration",
    ):
        """
        Initializes the vector field with various architectural parameters for
        concatenation and compression layers.

        :param key: JAX PRNGKey for random initialization.
        :type key: PRNGKeyArray
        :param n_neurons: List of integers specifying the number of neurons
            in each concatenation layer.
        :type n_neurons: list
        :param grid_size: Size of the input grid data. Required for Fourier
            space convolutions. Defaults to None.
        :type grid_size: Union[list, int], optional
        :param kernel_size: Size of the convolutional kernels. Defaults to 3.
        :type kernel_size: int, optional
        :param cell_size: Size of the simulation cell for E3 convolutions.
            Defaults to 1.0.
        :type cell_size: float, optional
        :param conv_irreps: Irreducible representations for E3 convolutions.
            Defaults to None.
        :type conv_irreps: Union[list, None], optional
        :param n_neurons_radial: Number of neurons in radial layers for E3
            convolutions. Defaults to [4].
        :type n_neurons_radial: list, optional
        :param n_neurons_lins: Number of neurons in linear layers for grid
            compression. Defaults to None.
        :type n_neurons_lins: Union[list, None], optional
        :param n_neurons_array: Number of neurons in linear layers for array
            compression. Defaults to None.
        :type n_neurons_array: Union[list, None], optional
        :param pooling_stride: Stride for pooling operations. Defaults to 2.
        :type pooling_stride: int, optional
        :param kernel_pooling_size: Kernel size for pooling operations.
            Defaults to 3.
        :type kernel_pooling_size: int, optional
        :param n_channels: Number of channels for ND convolutions.
            Defaults to None.
        :type n_channels: Union[list, None], optional
        :param conv_space: Specifies the convolution space ("configuration" or
            "fourier"). Defaults to "configuration".
        :type conv_space: str, optional
        :raises ValueError: If kernel_size is even for "configuration" space CNN
            or odd for "fourier" space CNN.
        :raises ValueError: If grid_size is not provided for "fourier" space
            convolutions with `conv_irreps`.
        """
        # Check the parity of the kernel
        if conv_space == "configuration" and kernel_size % 2 == 0:
            raise ValueError(
                "The kernel size must be odd for the configuration space cnn"
            )
        elif conv_space == "fourier" and kernel_size % 2 != 0:
            raise ValueError(
                "The kernel size must be even for the fourier space cnn"
            )

        # Get the size of the compressed grid and array
        if len(n_neurons_lins) != 0:
            compressed_grid_size = n_neurons_lins[-1]
        else:
            compressed_grid_size = 0
        if len(n_neurons_array) != 0:
            compressed_array_size = n_neurons_array[-1]
        else:
            compressed_array_size = 0

        # Get the number of layers
        n_layers = len(n_neurons) - 1

        # Split the key
        key_concat, key_compress_grid, key_compress_array = jrandom.split(
            key, 3
        )

        # Initialize the concatenation layers
        keys_concat = jrandom.split(key_concat, n_layers)
        self.concatenation_layers = []
        for i in range(n_layers):
            self.concatenation_layers.append(
                layers.concat_layer(
                    key=keys_concat[i],
                    in_size=n_neurons[i],
                    out_size=n_neurons[i + 1],
                    compressed_grid_size=compressed_grid_size,
                    compressed_array_size=compressed_array_size,
                )
            )

        # Auxiliary function to select the weights of eqx.nn.Linear
        def _select_weights_and_bias(
            layer: layers.concat_layer,
        ) -> tuple[Array, Array | None, Array, Array | None, Array]:
            return (
                layer.concat_data.weight,
                layer.concat_data.bias,
                layer.time_dilatation.weight,
                layer.time_dilatation.bias,
                layer.time_shift.weight,
            )

        # Function that returns the tree with zero weights
        def _zero_weights(
            layer: layers.concat_layer,
        ) -> tuple[Array, Array | None, Array, Array | None, Array]:
            return (
                jnp.zeros_like(layer.concat_data.weight),
                jnp.zeros_like(layer.concat_data.bias)
                if layer.concat_data.bias is not None
                else None,
                jnp.zeros_like(layer.time_dilatation.weight),
                jnp.zeros_like(layer.time_dilatation.bias)
                if layer.time_dilatation.bias is not None
                else None,
                jnp.zeros_like(layer.time_shift.weight),
            )

        # Set the weights of the last concatenation layer to zero
        self.concatenation_layers[-1] = eqx.tree_at(
            _select_weights_and_bias,
            self.concatenation_layers[-1],
            _zero_weights(self.concatenation_layers[-1]),
        )

        # Initialize the compression array layers
        self.compression_array_layers = []
        if compressed_array_size > 0:
            keys_compress_array = jrandom.split(key_compress_array, n_layers)
            for i in range(n_layers):
                self.compression_array_layers.append(
                    layers.compress_array(
                        key=keys_compress_array[i],
                        n_neurons_lins=n_neurons_array,
                    )
                )
        else:
            for i in range(n_layers):
                self.compression_array_layers.append(lambda _: jnp.array([]))

        # Initialize the compression grid layers
        self.compression_grid_layers = []
        if compressed_grid_size > 0:
            keys_compress_grid = jrandom.split(key_compress_grid, n_layers)
            for i in range(n_layers):
                if len(conv_irreps) == 0:
                    # Check if the n_channels is provided
                    if len(n_channels) == 0:
                        raise ValueError(
                            "The n_channels must be provided for the non-equivariant compression!"
                        )

                    # Check if the grid_size is provided
                    if len(grid_size) == 0:
                        raise ValueError(
                            "The grid_size must be provided for the non-equivariant compression!"
                        )

                    dimension = len(grid_size)
                    self.compression_grid_layers.append(
                        layers.compress_nd(
                            key=keys_compress_grid[i],
                            dimension=dimension,
                            kernel_size=kernel_size,
                            conv_channels=n_channels,
                            n_neurons_lins=n_neurons_lins,
                            pooling_stride=pooling_stride,
                            kernel_pooling_size=kernel_pooling_size,
                        )
                    )
                else:
                    if conv_space == "configuration":
                        self.compression_grid_layers.append(
                            layers.compress_3d_e3(
                                key=keys_compress_grid[i],
                                kernel_size=kernel_size,
                                cell_size=cell_size,
                                conv_irreps=conv_irreps,
                                n_neurons_lins=n_neurons_lins,
                                n_neurons_radial=n_neurons_radial,
                                pooling_stride=pooling_stride,
                                kernel_pooling_size=kernel_pooling_size,
                            )
                        )
                    elif conv_space == "fourier":
                        if len(grid_size) == 0:
                            raise ValueError(
                                "The grid_size must be provided for the fourier space convolutions!"
                            )

                        self.compression_grid_layers.append(
                            layers.compress_fourier_3d_e3(
                                key=keys_compress_grid[i],
                                grid_size=grid_size,
                                cell_size=cell_size,
                                conv_irreps=conv_irreps,
                                n_neurons_lins=n_neurons_lins,
                                n_neurons_radial=n_neurons_radial,
                                downsampling_factor=pooling_stride,
                            )
                        )
        else:
            for i in range(n_layers):
                self.compression_grid_layers.append(layers.no_compression())

    # Compute the kernels of the cnns
    def compute_kernels(self):
        """
        Computes the kernels for all convolutional layers within the vector field.
        This method iterates through the `convs` attribute (though it's not
        explicitly defined in the class, assuming it refers to internal
        convolutional components within compression layers if they exist).
        """
        for compression in self.compression_grid_layers:
            compression.compute_kernels()

    # Compress the array
    def compress_array(
        self, array: Float[Array, "array_size"]
    ) -> Float[Array, "N_concat_layers compressed_array_size"]:
        """
        Compresses the input array data using the initialized array compression layers.
        The compressed data is stored internally in `self.compressed_array_data`.

        :param array: The input array to be compressed.
        :type array: Float[Array, "array_size"]
        """
        compressed_array = []
        for compress_array_layer in self.compression_array_layers:
            compressed_array.append(compress_array_layer(array))

        return jnp.array(compressed_array)

    # Compress the grid
    def compress_grid(
        self,
        grid: Float[Array, "grid_size grid_size grid_size channel_size"],
    ) -> Float[Array, "N_concat_layers compressed_grid_size"]:
        """
        Compresses the input grid data using the initialized grid compression layers.
        The compressed data is stored internally in `self.compressed_grid_data`.

        :param grid: The input grid to be compressed, typically 3D with channels.
        :type grid: Float[Array, "grid_size grid_size grid_size channel_size"]
        """
        compressed_grid = []
        for compress_grid_layer in self.compression_grid_layers:
            compressed_grid.append(compress_grid_layer(grid))

        return jnp.array(compressed_grid)

    # Compute the vector field for a given vector and time
    def __call__(
        self,
        t: Float[Scalar, ""],
        theta: Float[Array, "theta_size"],
        args: Tuple[
            Float[Array, "N_concat_layers compressed_grid_size"],
            Float[Array, "N_concat_layers compressed_array_size"],
        ],
    ) -> Float[Array, "vector_field_size"]:
        """
        Computes the output of the vector field for a given time `t` and
        input vector `theta`.

        This method applies a series of concatenation layers, combining `theta`
        with the pre-compressed grid and array data, followed by a GELU activation.

        :param t: The current time step.
        :type t: float
        :param theta: The input vector (e.g., latent state) to the vector field.
        :type theta: Float[Array, "in_size"]
        :param args: Additional arguments, typically not used directly by this
            vector field but passed through by the ODE solver.
        :return: The transformed vector field output.
        :rtype: Float[Array, "in_size"]
        """
        # Unpack the arguments
        compressed_grid, compressed_array = args

        # Compute the concatenations
        for i in range(len(self.concatenation_layers) - 1):
            theta = self.concatenation_layers[i](
                t, theta, compressed_grid[i], compressed_array[i]
            )
            theta = jnn.gelu(theta)
        theta = self.concatenation_layers[-1](
            t, theta, compressed_grid[-1], compressed_array[-1]
        )

        return theta


# Class that computes the mean and std of the base distribution
class mean_std_layer(eqx.Module):
    concatenation_layer: eqx.nn.Linear
    compression_grid_layer: (
        layers.compress_nd
        | layers.compress_3d_e3
        | layers.compress_fourier_3d_e3
        | layers.no_compression
    )
    compression_array_layer: (
        layers.compress_array | Callable[[jnp.ndarray], jnp.ndarray]
    )

    # Initialize the vector field
    def __init__(
        self,
        key: PRNGKeyArray,
        n_neurons_out: int,
        grid_size: list[int] = [],
        kernel_size: int = 3,
        cell_size: float = 1.0,
        conv_irreps: list[str] = [],
        n_neurons_radial: list[int] = [4],
        n_neurons_lins: list[int] = [],
        n_neurons_array: list[int] = [],
        pooling_stride: int = 2,
        kernel_pooling_size: int = 3,
        n_channels: list[int] = [],
        conv_space: str = "configuration",
    ):
        """
        Initializes the vector field with various architectural parameters for
        concatenation and compression layers.

        :param key: JAX PRNGKey for random initialization.
        :type key: PRNGKeyArray
        :param n_neurons: List of integers specifying the number of neurons
            in each concatenation layer.
        :type n_neurons: list
        :param grid_size: Size of the input grid data. Required for Fourier
            space convolutions. Defaults to None.
        :type grid_sizFloat[Scalar, ""]e: Union[list, int], optional
        :param kernel_size: Size of the convolutional kernels. Defaults to 3.
        :type kernel_size: int, optional
        :param cell_size: Size of the simulation cell for E3 convolutions.
            Defaults to 1.0.
        :type cell_size: float, optional
        :param conv_irreps: Irreducible representations for E3 convolutions.
            Defaults to None.
        :type conv_irreps: Union[list, None], optional
        :param n_neurons_radial: Number of neurons in radial layers for E3
            convolutions. Defaults to [4].
        :type n_neurons_radial: list, optional
        :param n_neurons_lins: Number of neurons in linear layers for grid
            compression. Defaults to None.
        :type n_neurons_lins: Union[list, None], optional
        :param n_neurons_array: Number of neurons in linear layers for array
            compression. Defaults to None.
        :type n_neurons_array: Union[list, None], optional
        :param pooling_stride: Stride for pooling operations. Defaults to 2.
        :type pooling_stride: int, optional
        :param kernel_pooling_size: Kernel size for pooling operations.
            Defaults to 3.
        :type kernel_pooling_size: int, optional
        :param n_channels: Number of channels for ND convolutions.
            Defaults to None.
        :type n_channels: Union[list, None], optional
        :param conv_space: Specifies the convolution space ("configuration" or
            "fourier"). Defaults to "configuration".
        :type conv_space: str, optional
        :raises ValueError: If kernel_size is even for "configuration" space CNN
            or odd for "fourier" space CNN.
        :raises ValueError: If grid_size is not provided for "fourier" space
            convolutions with `conv_irreps`.
        """
        # Check the parity of the kernel
        if conv_space == "configuration" and kernel_size % 2 == 0:
            raise ValueError(
                "The kernel size must be odd for the configuration space cnn"
            )
        elif conv_space == "fourier" and kernel_size % 2 != 0:
            raise ValueError(
                "The kernel size must be even for the fourier space cnn"
            )

        # Get the size of the compressed grid and array
        if len(n_neurons_lins) != 0:
            compressed_grid_size = n_neurons_lins[-1]
        else:
            compressed_grid_size = 0
        if len(n_neurons_array) != 0:
            compressed_array_size = n_neurons_array[-1]
        else:
            compressed_array_size = 0

        # Split the key
        key_concat, key_compress_grid, key_compress_array = jrandom.split(
            key, 3
        )

        # Initialize the concatenation layers
        self.concatenation_layer = eqx.nn.Linear(
            compressed_grid_size + compressed_array_size,
            n_neurons_out,
            key=key_concat,
        )

        # Initialize the compression array layers
        if compressed_array_size > 0:
            self.compression_array_layer = layers.compress_array(
                key=key_compress_array, n_neurons_lins=n_neurons_array
            )
        else:
            self.compression_array_layer = lambda x: jnp.array([])

        # Initialize the compression grid layers
        if compressed_grid_size > 0:
            if len(conv_irreps) == 0:
                # Check if the n_channels is provided
                if len(n_channels) == 0:
                    raise ValueError(
                        "The n_channels must be provided for the non-equivariant compression!"
                    )

                # Check if the grid_size is provided
                if len(grid_size) == 0:
                    raise ValueError(
                        "The grid_size must be provided for the non-equivariant compression!"
                    )

                dimension = len(grid_size)
                self.compression_grid_layer = layers.compress_nd(
                    key=key_compress_grid,
                    dimension=dimension,
                    kernel_size=kernel_size,
                    conv_channels=n_channels,
                    n_neurons_lins=n_neurons_lins,
                    pooling_stride=pooling_stride,
                    kernel_pooling_size=kernel_pooling_size,
                )
            else:
                if conv_space == "configuration":
                    self.compression_grid_layer = layers.compress_3d_e3(
                        key=key_compress_grid,
                        kernel_size=kernel_size,
                        cell_size=cell_size,
                        conv_irreps=conv_irreps,
                        n_neurons_lins=n_neurons_lins,
                        n_neurons_radial=n_neurons_radial,
                        pooling_stride=pooling_stride,
                        kernel_pooling_size=kernel_pooling_size,
                    )
                elif conv_space == "fourier":
                    if len(grid_size) == 0:
                        raise ValueError(
                            "The grid_size must be provided for the fourier space convolutions!"
                        )

                    self.compression_grid_layer = (
                        layers.compress_fourier_3d_e3(
                            key=key_compress_grid,
                            grid_size=grid_size,
                            cell_size=cell_size,
                            conv_irreps=conv_irreps,
                            n_neurons_lins=n_neurons_lins,
                            n_neurons_radial=n_neurons_radial,
                            downsampling_factor=pooling_stride,
                        )
                    )
        else:
            self.compression_grid_layer = layers.no_compression()

    # Compute the kernels of the cnns
    def compute_kernels(self):
        """
        Computes the kernels for all convolutional layers within the vector field.
        This method iterates through the `convs` attribute (though it's not
        explicitly defined in the class, assuming it refers to internal
        convolutional components within compression layers if they exist).
        """
        self.compression_grid_layer.compute_kernels()

    # Compress the array
    def compress_array(
        self, array: Float[Array, "array_size"]
    ) -> Float[Array, "N_concat_layers compressed_array_size"]:
        """
        Compresses the input array data using the initialized array compression layers.
        The compressed data is stored internally in `self.compressed_array_data`.

        :param array: The input array to be compressed.
        :type array: Float[Array, "array_size"]
        """
        return self.compression_array_layer(array)

    # Compress the grid
    def compress_grid(
        self,
        grid: Float[Array, "grid_size grid_size grid_size channel_size"],
    ) -> Float[Array, "N_concat_layers compressed_grid_size"]:
        """
        Compresses the input grid data using the initialized grid compression layers.
        The compressed data is stored internally in `self.compressed_grid_data`.

        :param grid: The input grid to be compressed, typically 3D with channels.
        :type grid: Float[Array, "grid_size grid_size grid_size channel_size"]
        """
        return self.compression_grid_layer(grid)

    # Compute the vector field for a given vector and time
    def __call__(
        self,
        compressed_grid: Float[Array, "N_concat_layers compressed_grid_size"],
        compressed_array: Float[
            Array, "N_concat_layers compressed_array_size"
        ],
    ) -> Float[Array, "vector_field_size"]:
        """
        Computes the output of the vector field for a given time `t` and
        input vector `theta`.

        This method applies a series of concatenation layers, combining `theta`
        with the pre-compressed grid and array data, followed by a GELU activation.

        :param t: The current time step.
        :type t: float
        :param theta: The input vector (e.g., latent state) to the vector field.
        :type theta: Float[Array, "in_size"]
        :param args: Additional arguments, typically not used directly by this
            vector field but passed through by the ODE solver.
        :return: The transformed vector field output.
        :rtype: Float[Array, "in_size"]
        """
        # Compute the concatenations
        return jnn.tanh(
            self.concatenation_layer(
                jnp.hstack([compressed_grid, compressed_array])
            )
        )


# Define the class for the continuous normalizing flow
class cnf(eqx.Module):
    """
    Continuous Normalizing Flow (CNF) class that implements a neural ODE-based flow.
    """

    t0: float
    t1: float
    dt0: float
    theta_size: int
    vector_fields: list[vector_field_layer]
    mean_std: mean_std_layer

    def __init__(
        self,
        key: PRNGKeyArray,
        n_neurons: list[int],
        n_fields: int = 1,
        grid_size: list[int] = [],
        kernel_size: int = 3,
        cell_size: float = 1.0,
        conv_irreps: list[str] = [],
        n_neurons_radial: list[int] = [4],
        n_neurons_lins: list[int] = [],
        n_neurons_array: list[int] = [],
        pooling_stride: int = 2,
        kernel_pooling_size: int = 3,
        n_channels: list[int] = [],
        conv_space: str = "configuration",
        t0: float = 0.0,
        t1: float = 1.0,
        dt0: float = 0.1,
    ):
        """
        Initialize the CNF model.

        :param y_size: Dimension of the target space.
        :type y_size: int
        :param key: Random key for initialization.
        :type key: PRNGKeyArray
        :param num_blocks: Number of consecutive flow transformations.
        :type num_blocks: int, optional
        :param n_neurons: List of neurons for each layer in the vector field.
        :type n_neurons: list, optional
        :param n_neurons_mean_std: List of neurons for mean/std network.
        :type n_neurons_mean_std: list, optional
        :param x_size: Dimension of conditional input (0 for unconditional).
        :type x_size: Union[int, list], optional
        :param kernel_size: Size of convolutional kernels.
        :type kernel_size: int, optional
        :param n_neurons_radial: Neurons in radial layers for E3 convolutions.
        :type n_neurons_radial: list, optional
        :param pooling_stride: Stride for pooling operations.
        :type pooling_stride: int, optional
        :param kernel_pooling_size: Kernel size for pooling.
        :type kernel_pooling_size: int, optional
        :param n_channels: Channel sizes for ND convolutions.
        :type n_channels: list, optional
        :param compact_type: Type of compression ('standard' or 'equivariant').
        :type compact_type: str, optional
        :param conv_irreps: Irreps for E3 convolutions.
        :type conv_irreps: list, optional
        :param conv_space: Space for convolutions ('configuration' or 'fourier').
        :type conv_space: str, optional
        """
        # Check the dimensions of the input and output for the vector field
        if n_neurons[0] != n_neurons[-1]:
            raise ValueError(
                "The input and output dimensions must be the same for the vector field!"
            )
        self.theta_size = n_neurons[0]

        # Check if t1 > t0
        if t1 <= t0:
            raise ValueError("t1 must be greater than t0!")

        # Save parameters
        self.t0 = t0
        self.t1 = t1
        self.dt0 = dt0

        # Split the key
        keys = jrandom.split(key, n_fields + 1)

        # Initialize vector fields
        self.vector_fields = []
        for i in range(n_fields):
            self.vector_fields.append(
                vector_field_layer(
                    key=keys[i],
                    n_neurons=n_neurons,
                    grid_size=grid_size,
                    kernel_size=kernel_size,
                    cell_size=cell_size,
                    conv_irreps=conv_irreps,
                    n_neurons_radial=n_neurons_radial,
                    n_neurons_lins=n_neurons_lins,
                    n_neurons_array=n_neurons_array,
                    pooling_stride=pooling_stride,
                    kernel_pooling_size=kernel_pooling_size,
                    n_channels=n_channels,
                    conv_space=conv_space,
                )
            )

        # Initialize mean/std network
        self.mean_std = mean_std_layer(
            key=keys[-1],
            n_neurons_out=2 * n_neurons[0],
            grid_size=grid_size,
            kernel_size=kernel_size,
            cell_size=cell_size,
            conv_irreps=conv_irreps,
            n_neurons_radial=n_neurons_radial,
            n_neurons_lins=n_neurons_lins,
            n_neurons_array=n_neurons_array,
            pooling_stride=pooling_stride,
            kernel_pooling_size=kernel_pooling_size,
            n_channels=n_channels,
            conv_space=conv_space,
        )

    def compute_kernels(self):
        """
        Compute the kernels for all convolutional layers in each vector field of the CNF.
        """
        for vector_field in self.vector_fields:
            vector_field.compute_kernels()
        self.mean_std.compute_kernels()

    def _log_normal(
        self,
        theta: Float[Array, "theta_size"],
        compressed_grid: Float[Array, "N_concat_layers compressed_grid_size"],
        compressed_array: Float[
            Array, "N_concat_layers compressed_array_size"
        ],
    ) -> Float[Scalar, ""]:
        """
        Compute log probability under normal distribution.

        :param y: Input samples.
        :type y: Float[Array, "y_size"]
        :param x: Conditional input (optional).
        :type x: Float[Array, "x_size"], optional
        :return: Log probability of y under the base distribution.
        :rtype: float
        """
        # Compute the mean and std
        mu_std = self.mean_std(compressed_grid, compressed_array)
        mu, std = jnp.split(mu_std, 2, axis=-1)
        std = jnp.exp(std)

        return -0.5 * (
            theta.shape[-1] * jnp.log(2 * jnp.pi)
            + 2.0 * jnp.sum(jnp.log(std), axis=-1)
            + jnp.sum((theta - mu) ** 2 / std**2, axis=-1)
        )

    def _wrapper_func_trjac_approx(
        self,
        t: Float[Scalar, ""],
        y_trjac: tuple[Float[Array, "theta_size"], Float[Array, "1"]],
        args: tuple[
            Float[Array, "theta_size"],
            vector_field_layer,
            Float[Array, "N_concat_layers compressed_grid_size"],
            Float[Array, "N_concat_layers compressed_array_size"],
        ],
    ) -> tuple[Float[Array, "vector_field_size"], Float[Scalar, ""]]:
        """
        Wrapper function that computes vector field and trace Jacobian approximation.

        :param t: The current time step.
        :type t: Float
        :param y_trjac: A tuple containing the current state `y` and the trace Jacobian `trjac`.
        :type y_trjac: tuple
        :param args: A tuple containing `eps` (random perturbation) and the `vector_field` module.
        :type args: tuple
        :return: A tuple containing the computed vector field `f` and the updated trace Jacobian `trjac`.
        :rtype: tuple
        """
        # Unpack the arguments
        y, trjac = y_trjac
        eps, vector_field, compressed_grid, compressed_array = args

        # Compute vector field and the trace Jacobian
        def fn(
            y: Float[Array, "theta_size"],
        ) -> Float[Array, "vector_field_size"]:
            return vector_field(t, y, (compressed_grid, compressed_array))

        vjp_result = cast(
            Tuple[
                Float[Array, "theta_size"],
                Callable[
                    [Float[Array, "theta_size"]], Tuple[Float[Scalar, ""]]
                ],
            ],
            jax.vjp(fn, y),
        )
        (eps_dfdy,) = vjp_result[1](eps)
        trjac += jnp.sum(eps_dfdy * eps)

        return vjp_result[0], trjac

    def get_saveat(self, n_times: int = 1, reverse: bool = False) -> df.SaveAt:
        save_ts = jnp.linspace(self.t0, self.t1, n_times - 1, endpoint=False)
        save_ts = jnp.hstack([save_ts, self.t1])
        saveat = df.SaveAt(ts=save_ts[::-1] if reverse else save_ts)

        return saveat

    def get_logP(
        self,
        key: PRNGKeyArray,
        theta: Float[Array, "theta_size"],
        grid: Float[
            Array, "grid_size grid_size grid_size channel_size"
        ] = jnp.array([]),
        array: Float[Array, "array_size"] = jnp.array([]),
        saveat: df.SaveAt = df.SaveAt(ts=jnp.array([1.0])),
    ) -> tuple[Float[Array, "Ntimes theta_size"], Float[Array, "Ntimes 1"]]:
        """
        Compute log probability by solving ODE backward in time.

        :param y: Input samples.
        :type y: Float[Array, "y_size"]
        :param key: Random key.
        :type key: PRNGKeyArray
        :param save_ts: Time points to save during integration.
        :type save_ts: Float[Array, "Ntimes"]
        :param x: Conditional input (optional).
        :type x: Float[Array, "x_size"], optional
        :return: Tuple of (final samples, log probabilities, transformed x)
        :rtype: tuple
        """
        # Set up for solving ODE
        term = df.ODETerm(self._wrapper_func_trjac_approx) # pyright: ignore[reportArgumentType]
        solver = df.Tsit5()
        eps = jrandom.normal(key=key, shape=theta.shape)

        # Create the final theta array
        thetas_out: list[Float[Array, "n_times theta_size"]] = []

        # Run over all vector fields in reverse order
        delta_log_likelihood: Float[Scalar, ""] = jnp.array(0.0)
        for vector_field in reversed(self.vector_fields):
            # Compress the data
            compressed_grid = vector_field.compress_grid(grid)
            compressed_array = vector_field.compress_array(array)

            # Solve the ODE
            sol = df.diffeqsolve(
                term,
                solver,
                self.t1,
                self.t0,
                -self.dt0,
                (theta, delta_log_likelihood),
                (eps, vector_field, compressed_grid, compressed_array),
                saveat=saveat,
                stepsize_controller=df.PIDController(rtol=1e-4, atol=1e-4),
            )

            # Get the solution
            theta, delta_log_likelihood = cast(
                Tuple[
                    Float[Array, "n_times theta_size"],
                    Float[Scalar, ""],
                ],
                sol.ys,
            )

            # Save the thetas
            thetas_out.append(theta)
            theta = theta[-1, :]

        # Compute the base log normal probability
        compressed_grid = self.mean_std.compress_grid(grid)
        compressed_array = self.mean_std.compress_array(array)
        log_normal = self._log_normal(theta, compressed_grid, compressed_array)

        return (
            jnp.flip(jnp.vstack(thetas_out), axis=0),
            delta_log_likelihood[-1] + log_normal,
        )

    def _solve_ODE(
        self,
        theta: Float[Array, "theta_size"],
        grid: Float[Array, "grid_size"],
        array: Float[Array, "array_size"],
        saveat: df.SaveAt = df.SaveAt(ts=jnp.array([1.0])),
    ) -> Float[Array, "Ntimes y_size"]:
        """
        Solve ODE forward in time.

        :param y: Initial samples.
        :type y: Float[Array, "y_size"]
        :param Ntimes: Number of time points to return.
        :type Ntimes: int, optional
        :return: Samples at each time point.
        :rtype: Float[Array, "Ntimes y_size"]
        """
        # Set the solver
        solver = df.Tsit5()

        # Create the final theta array
        thetas_out: list[Float[Array, "n_times theta_size"]] = []

        # Run over all vector fields
        for vector_field in self.vector_fields:
            # Compress the data
            compressed_grid = vector_field.compress_grid(grid)
            compressed_array = vector_field.compress_array(array)

            sol = df.diffeqsolve(
                df.ODETerm(vector_field), # pyright: ignore[reportArgumentType]
                solver,
                self.t0,
                self.t1,
                self.dt0,
                theta,
                (compressed_grid, compressed_array),
                saveat=saveat,
                stepsize_controller=df.PIDController(rtol=1e-4, atol=1e-4),
            )

            # Get the solution
            theta = cast(Float[Array, "theta_size"], sol.ys)

            # Save the thetas
            thetas_out.append(theta)
            theta = theta[-1, :]

        return jnp.vstack(thetas_out)

    def sample(
        self,
        key: PRNGKeyArray,
        n_samples: int = 1000,
        grid: Float[
            Array, "grid_size grid_size grid_size channel_size"
        ] = jnp.array([]),
        array: Float[Array, "array_size"] = jnp.array([]),
        prior: Callable[[Array], float] = lambda _: 1,
        n_max: int = 100,
        n_times: int = 1,
    ) -> Float[Array, "Nsamples Ntimes theta_size"]:
        """
        Generate samples from the flow.

        :param key: Random key.
        :type key: PRNGKeyArray
        :param Nsamples: Number of samples to generate.
        :type Nsamples: int, optional
        :param x: Conditional input (optional).
        :type x: Float[Array, "x_size"], optional
        :param prior: Prior distribution function.
        :type prior: Callable, optional
        :param max_prior: Maximum prior value for rejection sampling.
        :type max_prior: float, optional
        :param Nmax: Maximum number of rejection sampling attempts.
        :type Nmax: int, optional
        :param Ntimes: Number of time points to return.
        :type Ntimes: int, optional
        :return: Generated samples.
        :rtype: Float[Array, "Nsamples ..."]
        """
        # Set up time points
        saveat = self.get_saveat(n_times=n_times, reverse=False)

        # Compute kernels
        self.compute_kernels()

        # Compute the mean and std of the base distribution
        compressed_grid = self.mean_std.compress_grid(grid)
        compressed_array = self.mean_std.compress_array(array)
        mu_std = self.mean_std(compressed_grid, compressed_array)
        mu, std = jnp.split(mu_std, 2, axis=-1)
        std = jnp.exp(std)

        # Generate samples
        n_out = 0
        count = 0
        theta_out = jnp.empty([0, n_times, self.theta_size])
        while n_out < n_samples and count < n_max:
            key, key_normal, key_uni = jrandom.split(key, 3)

            # Generate initial samples
            theta_ini = (
                jrandom.normal(key_normal, (n_samples, self.theta_size)) * std
                + mu
            )

            # Evolve ODE
            theta = jax.vmap(self._solve_ODE, in_axes=(0, None, None, None))(
                theta_ini, grid, array, saveat
            )

            # Apply prior and rejection sampling
            P_prior = jax.vmap(prior)(theta[:, -1, :])
            P_prior = P_prior / jnp.max(P_prior)

            eps = jrandom.uniform(key_uni, shape=(n_samples,))
            mask = eps < P_prior
            theta_out = jnp.vstack([theta_out, theta[mask, :, :]])
            n_out = theta_out.shape[0]
            count += 1

        return theta_out[:n_samples, :, :]


# Function used to create the mask the remove the kernels from the cnf
def get_mask(model: cnf):
    # Function that set the mask
    def _mask(path: Any, leaf: Any) -> bool:
        try:
            # r_grid from the conv_e3_layer
            # k2 from the conv_fourier_e3_layer
            # kernel from the pool_e3_layer
            name = path[-1].name
            if name == "r_grid" or name == "k2" or name == "kernel":
                return False
            else:
                return eqx.is_inexact_array(leaf)
        except AttributeError:
            try:
                # kernels from the conv_e3_layer and conv_fourier_e3_layer
                name = path[-2].name
                if name == "kernel":
                    return False
                else:
                    return eqx.is_inexact_array(leaf)
            except AttributeError:
                try:
                    # kernels from the conv_e3_layer and conv_fourier_e3_layer
                    name = path[-3].name
                    if name == "kernels":
                        return False
                    else:
                        return eqx.is_inexact_array(leaf)
                except AttributeError:
                    return eqx.is_inexact_array(leaf)

    return jtree.map_with_path(_mask, model)

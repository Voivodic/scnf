"""
This module implements some layers used in the construction of different types of continuous normalizing flows.
"""

# Import main libraries
import e3nn_jax as e3nn
import equinox as eqx

# Import jax
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from jaxtyping import Array, Float, Int, PRNGKeyArray, Union


# Convolutional layer [equivariant under E(3)]
class conv_e3_layer(eqx.Module):
    """
    This class implements a 3D convolutional layer that is equivariant to E(3).
    """

    phi: list
    r_grid: Float[Array, "kernel_size*kernel_size*kernel_size"]
    weights: list
    kernels: list
    kernel: list
    kernel_size: int
    stride: int

    # Initialize the class
    def __init__(
        self,
        key: PRNGKeyArray,
        irreps_in: e3nn.Irreps,
        irreps_out: e3nn.Irreps,
        kernel_size: int = 3,
        stride: int = 1,
        cell_size: int = 1.0,
        n_neurons_radial: list = [4, 4],
    ):
        """
        Initialize the equivariant convolutional layer.
        
        :param key: Key for the random number generator
        :type key: jax.random.PRNGKeyArray
        :param irreps_in: Array with the irreps of the input
        :type irreps_in: e3nn_jax.Irreps
        :param irreps_out: Array with the irreps of the output
        :type irreps_out: e3nn_jax.Irreps
        :param kernel_size: Size of the convolutional kernel
        :type kernel_size: int
        :param stride: Stride used in the convolutions
        :type stride: int
        :param cell_size: Size of each cell of the grid
        :type cell_size: float
        :param n_neurons_radial: Number of hidden neurons in the radial layer
        :type n_neurons_radial: list
        """
        # Check if the kernel size is odd
        if kernel_size % 2 == 0:
            raise ValueError("The kernel_size must be odd!")

        # Compute the positions of the and radial distances of the kernel
        self.kernel_size = kernel_size
        self.stride = stride
        kernel_side = (kernel_size - 1) / 2
        x, y, z = jnp.meshgrid(
            jnp.arange(-kernel_side, kernel_side + 1),
            jnp.arange(-kernel_side, kernel_side + 1),
            jnp.arange(-kernel_side, kernel_side + 1),
        )
        pos_grid = jnp.stack((x, y, z), axis=-1) * cell_size
        self.r_grid = jnp.sqrt(jnp.sum(jnp.power(pos_grid, 2), axis=-1))

        # Split the key for the radial and the radial kernels
        key_radial, key_angular = jrandom.split(key, 2)

        # Get informations about the input and output representations
        ls_in = irreps_in.ls
        ls_out = irreps_out.ls
        j_max = irreps_in.lmax + irreps_out.lmax

        # If the kernel is too small sets a smaller Jmax
        if j_max > 3 * kernel_side**2:
            j_max = int(3 * kernel_side**2)

        # Compute the spherical harmonics used
        yj = []
        for i in range(j_max + 1):
            yj.append(e3nn.sh(irreps_out=i, input=pos_grid, normalize=True))

        # Define the windows used to remove the high frequencies
        wj = []
        for i in range(j_max + 1):
            window = jnp.ones((kernel_size, kernel_size, kernel_size))
            window = window.at[self.r_grid < jnp.sqrt(i) * cell_size].set(0.0)
            wj.append(window)
        self.r_grid = self.r_grid.reshape([kernel_size**3, 1])

        # Compute Q times YJ for each possible J to be used
        self.kernels = []
        self.weights = []
        for jin in ls_in:
            tmpk = []
            tmpw = []
            for jout in ls_out:
                ttmpk = []
                ttmpw = []
                for J in range(abs(jin - jout), jin + jout + 1):
                    if J > j_max:
                        continue

                    # Compute the Clebsch Gordan coefficients
                    cg = jnp.array(
                        e3nn.clebsch_gordan(int(jin), int(jout), int(J))
                    ) * jnp.sqrt(2.0 * J + 1.0)

                    # Compute the outer product
                    ttmpk.append(
                        jnp.einsum(
                            "jilmn,lmn->jilmn",
                            jnp.einsum("ijk,lmnk->jilmn", cg, yj[J]),
                            wj[J],
                        )
                    )

                    # Initialize the weight for this angular kernel
                    key_angular, key_weight = jrandom.split(key_angular, 2)
                    ttmpw.append(jrandom.normal(key_weight))

                # Save the kernels and weights for this jout
                tmpk.append(jnp.array(ttmpk))
                tmpw.append(jnp.array(ttmpw))

            # Save the kernels and weights for this jin
            self.kernels.append(tmpk)
            self.weights.append(tmpw)

        # Define the MPL for the radial part
        depth = len(n_neurons_radial)
        keys_radial = jrandom.split(key_radial, depth + 1)
        self.phi = [eqx.nn.Linear(1, n_neurons_radial[0], key=keys_radial[0])]
        for i in range(depth - 1):
            self.phi.append(
                eqx.nn.Linear(
                    n_neurons_radial[i], n_neurons_radial[i + 1], key=keys_radial[i + 1]
                )
            )
        self.phi.append(eqx.nn.Linear(n_neurons_radial[depth], 1, key=keys_radial[-1]))

        # Set the initial kernel
        self.kernel = [0.0]

    # Pre-compute the kernel used in the convolutions
    def compute_kernel(self):
        """
        Compute the kernel used in the convolutions using the current weights.
        """
        # Compute the radial part
        phir = self.r_grid
        for layer in self.phi:
            phir = jnn.tanh(jax.vmap(layer)(phir))
        phir = phir.reshape(
            [1, 1, self.kernel_size, self.kernel_size, self.kernel_size]
        )

        # Compute the angular part
        kernel = jtu.tree_map_with_path(
            lambda _, x, y: jnp.einsum("a,aijlmn->ijlmn", x, y),
            self.weights,
            self.kernels,
        )
        kernel = jnp.hstack([jnp.vstack(x) for x in kernel])

        # Construct the final kernel
        self.kernel[0] = kernel * phir

    # Compute the layer for a given batch of inputs
    def __call__(self, x: Float[Array, "grid_size grid_size grid_size channel_size"]):
        """
        Compute the convolutional layer for a given input.

        :param x: Input array
        :type x: jax.numpy.array

        :return: Input convoluted with the kernel
        :rtype: jax.numpy.array
        """
        # Compute the convolution of the kernel with the input array
        return jax.lax.conv_general_dilated(
            lhs=jnp.expand_dims(x, axis=0),
            rhs=self.kernel[0],
            window_strides=[self.stride, self.stride, self.stride],
            padding="VALID",
            dimension_numbers=("NHWDC", "OIHWD", "NHWDC"),
        )[0]


# Define the layer that compress the information in the grid in an invariant way
class compress_3d_e3(eqx.Module):
    """
    Compress a 3D grid with a result invariant to E(3)
    """

    convs: list
    lins: list
    irreps: list
    pool: eqx.nn.Pool
    pad_size: tuple

    # Initialize the parameters of the class
    def __init__(
        self,
        key: PRNGKeyArray,
        kernel_size: int = 3,
        stride_size: int = 1,
        cell_size: float = 1.0,
        conv_irreps: list = [
            e3nn.Irreps("1x0e"),
            e3nn.Irreps("5x0e"),
            e3nn.Irreps("10x0e+1x2e"),
            e3nn.Irreps("20x0e+1x1o+2x2e"),
            e3nn.Irreps("64x0e"),
        ],
        n_neurons_lins: list = [64, 32, 16, 8],
        n_neurons_radial: list = [4, 4],
        pooling_size: int = 2,
    ):
        """
        Initialize the class.

        :param key: Key for random number generation
        :type key: PRNGKeyArray
        :param kernel_size: Size of the convolutional kernels
        :type kernel_size: int
        :param stride_size: Stride for the convolutions
        :type stride_size: int
        :param cell_size: Size of the cell in the input grid
        :type cell_size: float
        :param conv_irreps: Irreps for the convolutional layers
        :type conv_irreps: list
        :param n_neurons_lins: Number of neurons for the linear layers
        :type n_neurons_lins: list
        :param n_neurons_radial: Number of neurons for the radial part
        :type n_neurons_radial: list
        :param pooling_size: Size of the pooling layer
        :type pooling_size: int
        """
        # Set the keys
        key_conv, key_lin = jrandom.split(key, 2)
        Nconv = len(conv_irreps)
        Nlin = len(n_neurons_lins)
        keys_conv = jrandom.split(key_conv, Nconv - 1)
        keys_lin = jrandom.split(key_lin, Nlin - 1)

        # Check if all dimensions are even
        kernel_side = (kernel_size - 1) // 2
        self.pad_size = (
            (kernel_side, kernel_side),
            (kernel_side, kernel_side),
            (kernel_side, kernel_side),
            (0, 0),
        )

        # Construct the convolutional layers
        self.irreps = conv_irreps
        self.convs = []
        for i in range(Nconv - 1):
            self.convs.append(
                conv_e3_layer(
                    irreps_in=conv_irreps[i],
                    irreps_out=conv_irreps[i + 1],
                    kernel_size=kernel_size,
                    stride=stride_size,
                    key=keys_conv[i],
                    cell_size=cell_size,
                    n_neurons_radial=n_neurons_radial,
                )
            )

        # Construct the linear layers
        self.lins = []
        for i in range(Nlin - 1):
            self.lins.append(
                eqx.nn.Linear(n_neurons_lins[i], n_neurons_lins[i + 1], key=keys_lin[i])
            )

        # Set the pooling layer
        self.pool = eqx.nn.AvgPool3d(
            kernel_size=pooling_size, stride=pooling_size, padding=0
        )

    # Pre-compute the kernels of all conv layers
    def compute_kernels(self):
        """
        Compute the kernels of all convolutional layers.
        """
        for conv in self.convs:
            conv.compute_kernel()

    # Compress the input grids
    def __call__(self, x: Float[Array, "grid_size grid_size grid_size channel_size"]):
        """
        Compress the given 3D grid using the equivariant convolutional layers.

        :param x: Input 3D grids
        :type x: jax.numpy.array

        :return: Compressed 3D grid
        :rtype: jax.numpy.array
        """
        # Apply the conv layers
        for i, conv in enumerate(self.convs):
            x = jnp.pad(x, pad_width=self.pad_size, mode="wrap")
            x = conv(x)
            x = e3nn.gate(
                e3nn.IrrepsArray(
                    irreps=self.irreps[i + 1], array=jnp.transpose(x, (3, 0, 1, 2))
                ),
                even_act=jnn.tanh,
                even_gate_act=jnn.tanh,
                normalize_act=True,
            )
            x = self.pool(x)

        # Flatten the data
        x = jnp.mean(x, axis=(0, 1, 2))

        # Apply the linear layers
        for i in range(len(self.lins) - 1):
            x = jnn.relu(self.lins[i](x))
        x = self.lins[-1](x)

        return x


# Define the layer that compress the information in the grid with dimension n
class compress_nd(eqx.Module):
    """
    Compress a ND grid
    """

    convs: list
    lins: list
    pool: eqx.nn.Pool
    pad_size: tuple

    # Initialize the parameters of the class
    def __init__(
        self,
        key: PRNGKeyArray,
        dimension: int = 3,
        kernel_size: int = 3,
        stride_size: int = 1,
        conv_channels: list = [1, 5, 15, 33, 64],
        n_neurons_lins: list = [64, 32, 16, 8],
        pooling_size: int = 2,
    ):
        """
        Initialize the class.

        :param dimension: The dimension of the grid to be compressed
        :type dimension: int
        :param key: Key for random number generation
        :type key: PRNGKeyArray
        :param kernel_size: Size of the convolutional kernels
        :type kernel_size: int
        :param stride_size: Stride for the convolutions
        :type stride_size: int
        :param conv_channels: Number of channels for the convolutional layers
        :type conv_irreps: list
        :param n_neurons_lins: Number of neurons for the linear layers
        :type n_neurons_lins: list
        :param pooling_size: Size of the pooling layer
        :type pooling_size: int
        """
        # Set the keys
        key_conv, key_lin = jrandom.split(key, 2)
        Nconv = len(conv_channels)
        Nlin = len(n_neurons_lins)
        keys_conv = jrandom.split(key_conv, Nconv - 1)
        keys_lin = jrandom.split(key_lin, Nlin - 1)

        # Compute the pad sizes for the periodic boundary conditions
        kernel_side = (kernel_size - 1) // 2
        self.pad_size = []
        for i in range(dimension):
            self.pad_size.append((kernel_side, kernel_side))
        self.pad_size.append((0, 0))

        # Construct the convolutional layers
        self.convs = []
        for i in range(Nconv - 1):
            self.convs.append(
                eqx.nn.Conv(
                    num_spatial_dims=dimension,
                    in_channels=conv_channels[i],
                    out_channels=conv_channels[i + 1],
                    kernel_size=kernel_size,
                    stride=stride_size,
                    padding=0,
                    key=keys_conv[i],
                )
            )

        # Construct the linear layers
        self.lins = []
        for i in range(Nlin - 1):
            self.lins.append(
                eqx.nn.Linear(n_neurons_lins[i], n_neurons_lins[i + 1], key=keys_lin[i])
            )

        # Set the pooling layer
        self.pool = eqx.nn.AvgPool(
            num_spatial_dims=dimension,
            kernel_size=pooling_size,
            stride=pooling_size,
            padding=0,
        )

    # Compress the input grids
    def __call__(self, x: Float[Array, "grid_size^dimension num_channels"]):
        """
        Compress the given ND grid using the convolutional layers.

        :param x: Input ND grid
        :type x: jax.numpy.array

        :return: Compressed 3D grid
        :rtype: jax.numpy.array
        """
        # Apply the conv layers
        for conv in self.convs:
            x = jnp.pad(x, pad_width=self.pad_size, mode="wrap")
            x = conv(x)
            x = jnn.tanh(x)
            x = self.pool(x)

        # Flatten the data
        x = jnp.mean(x, axis=(0, 1, 2))

        # Apply the linear layers
        for i in range(len(self.lins) - 1):
            x = jnn.relu(self.lins[i](x))
        x = self.lins[-1](x)

        return x


# Define the layer that compress the information in an array without convolutions
class compress_array(eqx.Module):
    """
    Compress an array
    """

    lins: list

    # Initialize the parameters of the class
    def __init__(
        self,
        key: PRNGKeyArray,
        n_neurons_lins: list = [3, 16, 16, 3],
    ):
        """
        Initialize the class.

        :param key: Key for random number generation
        :type key: PRNGKeyArray
        :param n_neurons_lins: Number of neurons for the linear layers
        :type n_neurons_lins: list
        """
        # Set the keys
        Nlin = len(n_neurons_lins)
        keys_lin = jrandom.split(key, Nlin - 1)

        # Construct the linear layers
        self.lins = []
        for i in range(Nlin - 1):
            self.lins.append(
                eqx.nn.Linear(n_neurons_lins[i], n_neurons_lins[i + 1], key=keys_lin[i])
            )

    # Compress the input grids
    def __call__(self, x: Float[Array, "grid_size^dimension num_channels"]):
        """
        Compress the given array using the linear layers.

        :param x: Input array
        :type x: jax.numpy.array

        :return: Compressed array
        :rtype: jax.numpy.array
        """
        # Apply the linear layers
        for i in range(len(self.lins) - 1):
            x = jnn.relu(self.lins[i](x))
        x = self.lins[-1](x)

        return x


# Define the layer that concatenate the conditionals (random choice from FFJORD)
class concat_layer(eqx.Module):
    """
    Concatenate the parameters with time and any other conditionals.
    """

    concat_layer: eqx.nn.Linear
    time_dilatation: eqx.nn.Linear
    time_shift: eqx.nn.Linear
    compress_x: eqx.Module
    compressed_x: list

    # Initialize the parameters of the layer
    def __init__(
        self,
        key: PRNGKeyArray,
        in_size: int,
        out_size: int,
        dimension: int = 3,
        kernel_size: int = 3,
        stride_size: int = 1,
        cell_size: float = 1.0,
        conv_irreps: Union[list, None] = None,
        n_neurons_lins: Union[list, None] = None,
        n_neurons_radial: list = [4],
        pooling_size: int = 2,
        n_channels: Union[list, None] = None,
    ):
        """
        Initialize the class.

        :param key: Key for random number generation
        :type key: PRNGKeyArray
        :param in_size: Size of the input array
        :type in_size: int
        :param out_size: Size of the output array
        :type out_size: int
        :param dimension: Dimension of the grid to be compressed (when applicable)
        :type dimension: int
        :param kernel_size: Size of the convolutional kernels (when applicable)
        :type kernel_size: int
        :param stride_size: Stride for the convolutions (when applicable)
        :type stride_size: int
        :param cell_size: Size of the cell in the input grid (when applicable)
        :type cell_size: float
        :param conv_irreps: Irreps for the convolutional layers (when applicable)
        :type conv_irreps: Union[list, None]
        :param n_neurons_lins: Number of neurons for the linear layers (when applicable)
        :type n_neurons_lins: Union[list, None]
        :param n_neurons_radial: Number of neurons for the radial part (when applicable)
        :type n_neurons_radial: list
        :param pooling_size: Size of the pooling layer (when applicable)
        :type pooling_size: int
        :param n_channels: Number of chennels of each layer of the convolutions (when applicable)
        :type n_channels: Union[list, None] 
        """

        # Set the keys
        key_concat, key_dilat, key_shift, key_compress = jrandom.split(key, 4)

        # Get the size of the compacted Array
        if n_neurons_lins is not None:
            compressed_size = n_neurons_lins[-1]
        else:
            compressed_size = 0

        # Define the layer that concatenate everything
        self.concat_layer = eqx.nn.Linear(
            in_size + compressed_size, out_size, key=key_concat
        )

        # Define the layers that transform the time coordinate
        self.time_dilatation = eqx.nn.Linear(1, out_size, key=key_dilat)
        self.time_shift = eqx.nn.Linear(1, out_size, use_bias=False, key=key_shift)

        # Define the layer that compress the information in the conditionals
        if compressed_size > 0:
            if conv_irreps is None and n_channels is None:
                self.compress_x = compress_array(
                    key=key_compress, n_neurons_lins=n_neurons_lins
                )
            else:
                if conv_irreps is None:
                    self.compress_x = compress_nd(
                        key=key_compress,
                        dimension=dimension,
                        kernel_size=kernel_size,
                        stride_size=stride_size,
                        conv_channels=n_channels,
                        n_neurons_lins=n_neurons_lins,
                        pooling_size=pooling_size,
                    )
                else:
                    self.compress_x = compress_3d_e3(
                        key=key_compress,
                        kernel_size=kernel_size,
                        stride_size=stride_size,
                        cell_size=cell_size,
                        conv_irreps=conv_irreps,
                        n_neurons_lins=n_neurons_lins,
                        n_neurons_radial=n_neurons_radial,
                        pooling_size=pooling_size,
                    )

        # Array to keep the compressed conditional information on [0]
        self.compressed_x = compressed_size * [0.0]

    # Compress the conditional information and save it
    def compress_x(self, x: Float[Array, "x_size num_channels"]):
        """
        Compress the grid/array and save the information

        :param x: Grid or array to be compressed
        :type x: jax.numpy.array
        """
        self.compressed_x[0] = jnn.tanh(self.compress_x(x))

    # Concatenate the time and the conditional information with the current vector
    def __call__(self, t: float, y: Float[Array]):
        """
        Apply the layer to concatenate the input array with the time and the compressed conditionals

        :param t: Time in the ODE
        :type t: float
        :param y: Array in the ODE
        :type y: jax.numpy.array

        :return: The input array concatenated with the time and conditional
        :rtype: jax.numpy.array
        """
        y_stacked = jnp.hstack([y, self.compressed_x[0]])
        y = self.concat_layer(y_stacked) * jnn.sigmoid(
            self.time_dilatation(t)
        ) + self.time_shift(t)

        return y

"""
This module implements some layers used in the construction of different types of continuous normalizing flows.
"""

# Import general libreries
from typing import List

import e3nn_jax as e3nn
import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float, List, PRNGKeyArray


# Class with one equivariant under E(3) layer
class conv_e3_layer(eqx.Module):
    """
    This class implements a 3D convolutional layer that is equivariant to E(3).
    """

    phi: List
    r_grid: Float[Array, "kernel_size*kernel_size*kernel_size"]
    weights: List
    kernels: List
    Kernel: list
    kernel_size: int
    stride: int
    irreps_in: e3nn.Irreps
    irreps_out: e3nn.Irreps

    # Initialize the class
    def __init__(
        self,
        irreps_in: e3nn.Irreps,
        irreps_out: e3nn.Irreps,
        key: PRNGKeyArray,
        kernel_size: int = 3,
        stride: int = 1,
        cell_size: int = 1.0,
        Nneurons_radial: list = [4, 4],
    ):
        """
        Initialize the equivariant convolutional layer.

        :param irreps_in: Array with the irreps of the input
        :type irreps_in: e3nn_jax.Irreps
        :param irreps_out: Array with the irreps of the output
        :type irreps_out: e3nn_jax.Irreps
        :param key: Key for the random number generator
        :type key: jax.random.PRNGKeyArray
        :param kernel_size: Size of the convolutional kernel
        :type kernel_size: int
        :param stride: Stride used in the convolutions
        :type stride: int
        :param cell_size: Size of each cell of the grid
        :type cell_size: float
        :param Nneurons_radial: Number of hidden neurons in the radial layer
        :type Nneurons_radial: list
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
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
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
        depth = len(Nneurons_radial)
        keys_radial = jrandom.split(key_radial, depth + 1)
        self.phi = [eqx.nn.Linear(1, Nneurons_radial[0], key=keys_radial[0])]
        for i in range(depth - 1):
            self.phi.append(
                eqx.nn.Linear(
                    Nneurons_radial[i], Nneurons_radial[i + 1], key=keys_radial[i + 1]
                )
            )
        self.phi.append(eqx.nn.Linear(Nneurons_radial[depth], 1, key=keys_radial[-1]))

        # Set the initial kernel
        self.Kernel = [0.0]

    # Pre-compute the kernel used in the convolutions
    def compute_kernel(self):
        """
        Compute the kernel used in the convolutions using the current weights.
        """
        # Compute the radial part
        phir = self.r_grid
        for layer in self.phi:
            phir = jax.nn.tanh(jax.vmap(layer)(phir))
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
        self.Kernel[0] = kernel * phir

    # Compute the layer for a given input
    def __call__(self, x: e3nn.IrrepsArray):
        """
        Compute the convolutional layer for a given input.

        :param x: Input IrrepsArray
        :type x: e3nn_jax.IrrepsArray

        :return: Output IrrepsArray of the input convoluted with the kernel
        :rtype: e3nn_jax.IrrepsArray
        """
        # Check the inputs
        if x.irreps != self.irreps_in:
            raise ValueError(
                "The irreps of the input is incompatile with the one used to initialize the class!"
            )

        # Take the input array
        y = x.array

        # Compute the convolution of the kernel with the input array
        y = jax.lax.conv_general_dilated(
            lhs=jnp.expand_dims(y, axis=0),
            rhs=self.Kernel[0],
            window_strides=[self.stride, self.stride, self.stride],
            padding="VALID",
            dimension_numbers=("NHWDC", "OIHWD", "NHWDC"),
        )

        return e3nn.IrrepsArray(self.irreps_out, y[0])


# Define the layer that compress the information in the grid in an invariant way
class compress_e3_grid(eqx.Module):
    """
    Compress a 3D grid with a result invariant to E(3)
    """
    convs: List
    lins: List
    pool: eqx.nn.Pool
    pad_size: tuple

    # Initialize the parameters of the class
    def __init__(
        self,
        key: PRNGKeyArray,
        kernel_size: int = 3,
        stride: int = 1,
        cell_size: float = 1.0,
        conv_irreps: list = [
            e3nn.Irreps("1x0e"),
            e3nn.Irreps("5x0e"),
            e3nn.Irreps("10x0e+1x2e"),
            e3nn.Irreps("20x0e+1x1o+2x2e"),
            e3nn.Irreps("64x0e"),
        ],
        Nneurons_conv_lins: list = [64, 32, 16, 8],
        Nneurons_conv_radial: list = [4, 4],
        pooling_size: int = 2,
    ):
        """
        Initialize the class.

        :param key: Key for random number generation
        :type key: PRNGKeyArray
        :param kernel_size: Size of the convolutional kernels
        :type kernel_size: int
        :param stride: Stride for the convolutions
        :type stride: int
        :param cell_size: Size of the cell in the input grid
        :type cell_size: float
        :param conv_irreps: Irreps for the convolutional layers
        :type conv_irreps: list
        :param Nneurons_conv_lins: Number of neurons for the linear layers 
        :type Nneurons_conv_lins: list
        :param Nneurons_conv_radial: Number of neurons for the radial part
        :type Nneurons_conv_radial: list
        :param pooling_size: Size of the pooling layer
        :type pooling_size: int
        """
        # Set the keys
        key_conv, key_lin = jrandom.split(key, 2)
        Nconv = len(conv_irreps)
        Nlin = len(Nneurons_conv_lins)
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
        self.convs = []
        for i in range(Nconv - 1):
            self.convs.append(
                conv_e3_layer(
                    irreps_in=conv_irreps[i],
                    irreps_out=conv_irreps[i + 1],
                    kernel_size=kernel_size,
                    stride=stride,
                    key=keys_conv[i],
                    cell_size=cell_size,
                    Nneurons_radial=Nneurons_conv_radial,
                )
            )

        # Construc the linear layers
        self.lins = []
        for i in range(Nlin - 1):
            self.lins.append(
                eqx.nn.Linear(
                    Nneurons_conv_lins[i], Nneurons_conv_lins[i + 1], key=keys_lin[i]
                )
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

    # Return the compressed x
    def __call__(self, grid: Float[Array, "(Nchannels)+grid_size"]):
        """
        Compress the given 3D grid using the convolutional layers.

        :param grid: Input 3D grid
        :type grid: Float[Array, "(Nchannels)+grid_size"]

        :return: Compressed 3D grid
        :rtype: Float[Array, "compression_size"]
        """
        # Expand x one dimension to take into account the channels and corvert to IrrepsArray
        x_irreps = e3nn.Irreps("1x0e") * grid.shape[0]
        x = jnp.transpose(grid.array, (1, 2, 3, 0))

        # Apply the conv layers
        for conv in self.convs:
            x = e3nn.IrrepsArray(
                x_irreps, jnp.pad(x, pad_width=self.pad_size, mode="wrap")
            )
            x = conv(x)
            x = e3nn.gate(
                x, even_act=jax.nn.tanh, even_gate_act=jax.nn.tanh, normalize_act=True
            )
            x_irreps = x.irreps
            x = jnp.transpose(
                self.pool(jnp.transpose(x.array, (3, 0, 1, 2))), (1, 2, 3, 0)
            )

        # Flatten the data
        x = jnp.mean(x, axis=(0, 1, 2))

        # Apply the linear layers
        for i in range(len(self.lins) - 1):
            x = jax.nn.relu(self.lins[i](x))
        x = self.lins[-1](x)

        return x

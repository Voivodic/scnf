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
from jaxtyping import Array, Float, Key, Union


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
        key: Key,
        irreps_in: e3nn.Irreps,
        irreps_out: e3nn.Irreps,
        kernel_size: int = 3,
        stride: int = 1,
        cell_size: int = 1.0,
        n_neurons_radial: list = [4, 4],
    ):
        """
        Initialize the equivariant convolutional layer.

        :param key: Key for the random number generator.
        :type key: Key
        :param irreps_in: Irreps of the input features.
        :type irreps_in: e3nn_jax.Irreps
        :param irreps_out: Irreps of the output features.
        :type irreps_out: e3nn_jax.Irreps
        :param kernel_size: Size of the convolutional kernel (must be odd).
        :type kernel_size: int
        :param stride: Stride used in the convolutions.
        :type stride: int
        :param cell_size: Size of each cell of the grid.
        :type cell_size: float
        :param n_neurons_radial: Number of hidden neurons in the radial layer.
        :type n_neurons_radial: list
        :raises ValueError: If `kernel_size` is an even number.
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
        self.phi.append(
            eqx.nn.Linear(n_neurons_radial[depth - 1], 1, key=keys_radial[-1])
        )

        # Set the initial kernel
        self.kernel = [0.0]

    # Pre-compute the kernel used in the convolutions
    def compute_kernel(self):
        """
        Compute the kernel used in the convolutions using the current weights.
        This method should be called to update the kernel after any changes to the layer's weights.
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

        :param x: Input array representing the 3D grid with dimensions (grid_size, grid_size, grid_size, channel_size).
        :type x: jax.numpy.array

        :return: Convolved output array with dimensions adjusted based on padding and stride.
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


# Convolutional layer [equivariant under E(3)] computed in Fourier space
class conv_fourier_e3_layer(eqx.Module):
    """
    This class implements a 3D convolutional layer that is equivariant to E(3).
    The convolution is computed in Fourier space.
    """

    phi: list
    k2: Float[Array, "kernel_size*kernel_size*kernel_size"]
    weights: list
    kernels: list
    kernel: list
    mask: Float[Array, "grid_size grid_size grid_size"]
    kernel_size: int

    # Initialize the class
    def __init__(
        self,
        key: Key,
        grid_size: list,
        irreps_in: e3nn.Irreps,
        irreps_out: e3nn.Irreps,
        cell_size: int = 1.0,
        kernel_size: int = 4,
        n_neurons_radial: list = [4, 4],
    ):
        """
        Initialize the equivariant convolutional layer.

        :param key: Key for the random number generator.
        :type key: Key
        :param grid_size: Dimensions of the input grid (e.g., [Nx, Ny, Nz]).
        :type grid_size: list
        :param irreps_in: Irreps of the input features.
        :type irreps_in: e3nn_jax.Irreps
        :param irreps_out: Irreps of the output features.
        :type irreps_out: e3nn_jax.Irreps
        :param cell_size: Size of each cell of the grid.
        :type cell_size: float
        :param kernel_size: Size of the convolutional kernel (must be even).
        :type kernel_size: int
        :param n_neurons_radial: Number of hidden neurons in the radial layer.
        :type n_neurons_radial: list
        :raises ValueError: If `kernel_size` is an odd number.
        """
        # Check tha the kernel size is EnvironmentError
        if kernel_size % 2 != 0:
            raise ValueError("The kernel_size must be even!")

        # Compute the size of the grid
        L = []
        for i in range(3):
            L.append(cell_size * grid_size[i])

        # Compute the positions of the and radial distances of the kernel
        x, y, z = jnp.meshgrid(
            jnp.roll(
                jnp.arange(-grid_size[0] // 2, grid_size[0] // 2), -grid_size[0] // 2
            ),
            jnp.roll(
                jnp.arange(-grid_size[1] // 2, grid_size[1] // 2), -grid_size[1] // 2
            ),
            jnp.hstack([jnp.arange(0, grid_size[2] // 2), -grid_size[2] // 2]),
        )
        kvec = jnp.stack((x, y, z), axis=-1)

        # Compute the mask
        kernel_side = kernel_size // 2
        self.mask = (
            ((-kernel_side <= kvec[:, :, :, 0]) & (kvec[:, :, :, 0] < kernel_side))
            & ((-kernel_side <= kvec[:, :, :, 1]) & (kvec[:, :, :, 1] < kernel_side))
            & ((-kernel_side <= kvec[:, :, :, 2]) & (kvec[:, :, :, 2] <= kernel_side))
        )
        self.kernel_size = kernel_size

        # Put physical units in the k grid
        kvec = kvec.astype(jnp.float32)
        kvec = kvec.at[:, :, :, 0].set(kvec[:, :, :, 0] * 2.0 * jnp.pi / L[0])
        kvec = kvec.at[:, :, :, 1].set(kvec[:, :, :, 1] * 2.0 * jnp.pi / L[1])
        kvec = kvec.at[:, :, :, 2].set(kvec[:, :, :, 2] * 2.0 * jnp.pi / L[2])
        kvec = kvec[self.mask, :].reshape(
            [kernel_size, kernel_size, kernel_size // 2 + 1, 3]
        )
        k2 = jnp.sum(jnp.power(kvec, 2), axis=-1)
        k2_max = (
            1.01
            * kernel_side**2
            * 4.0
            * jnp.pi**2
            / jnp.power(L[0] * L[1] * L[2], 2.0 / 3.0)
        )
        mask_r = k2 <= k2_max
        self.k2 = k2.reshape([jnp.prod(jnp.array(k2.shape)), 1])

        # Split the key for the radial and the radial kernels
        key_radial, key_angular = jrandom.split(key, 2)

        # Get informations about the input and output representations
        ls_in = irreps_in.ls
        ls_out = irreps_out.ls
        j_max = irreps_in.lmax + irreps_out.lmax

        # Compute the spherical harmonics used
        yj = []
        for i in range(j_max + 1):
            yj.append(e3nn.sh(irreps_out=i, input=kvec, normalize=True))
            yj[-1] = yj[-1].at[~mask_r, :].set(0.0)

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
                    # Compute the Clebsch Gordan coefficients
                    cg = jnp.array(
                        e3nn.clebsch_gordan(int(jin), int(jout), int(J))
                    ) * jnp.sqrt(2.0 * J + 1.0)

                    # Compute the outer product
                    ttmpk.append(
                        jnp.einsum("ijk,lmnk->jilmn", cg, yj[J]),
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
        self.phi.append(
            eqx.nn.Linear(n_neurons_radial[depth - 1], 1, key=keys_radial[-1])
        )

        # Set the initial kernel
        self.kernel = [0.0]

    def compute_kernel(self):
        """
        Compute the kernel used in the convolutions using the current weights.
        """
        # Compute the radial part
        phir = self.k2
        for layer in self.phi:
            phir = jnn.tanh(jax.vmap(layer)(phir))
        phir = phir.reshape(
            [1, 1, self.kernel_size, self.kernel_size, self.kernel_size // 2 + 1]
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

        :param x: Input array representing the 3D grid with dimensions (grid_size, grid_size, grid_size, channel_size).
        :type x: jax.numpy.array
        :returns: Convolved output array in configuration space.
        :rtype: jax.numpy.array
        """
        # Transform the input to Fourier space
        n_channels = x.shape[-1]
        x = jnp.fft.rfftn(x, axes=(0, 1, 2))
        x = x[self.mask, :].reshape(
            [self.kernel_size, self.kernel_size, self.kernel_size // 2 + 1, n_channels]
        )

        # Apply the kernel
        x = jnp.einsum("ijlmn,lmnj->lmni", self.kernel[0], x)

        # Transform back to configuration space
        x = jnp.fft.irfftn(
            x,
            axes=(0, 1, 2),
            s=[self.kernel_size, self.kernel_size, self.kernel_size],
        )

        return x


# Pooling layer [equivariant under E(3)]
class pool_e3_layer(eqx.Module):
    """This class implements a 3D pooling layer that is equivariant to E(3)."""

    kernel_size: int
    stride: int
    kernel: Float[Array, "kernel_size kernel_size kernel_size"]

    def __init__(
        self,
        kernel_size: int = 3,
        stride: int = 2,
        cell_size: float = 1.0,
        kernel_type: str = "exp",
    ):
        """Initialize the 3D pooling layer.

        :param kernel_size: Size of the pooling kernel.
        :type kernel_size: int
        :param stride: Stride used in the pooling.
        :type stride: int
        :param cell_size: Size of each cell of the grid.
        :type cell_size: float
        :param kernel_type: Type of kernel to use ('exp', 'gaussian', or 'const').
        :type kernel_type: str
        """
        # Set the parameters
        self.kernel_size = kernel_size
        self.stride = stride

        # Compute the kernel
        kernel_side = (kernel_size - 1) // 2
        if kernel_type == "exp":
            x, y, z = jnp.meshgrid(
                jnp.arange(-kernel_side, kernel_side + 1),
                jnp.arange(-kernel_side, kernel_side + 1),
                jnp.arange(-kernel_side, kernel_side + 1),
            )
            pos_grid = jnp.stack((x, y, z), axis=-1) * cell_size
            r_grid = jnp.sqrt(jnp.sum(jnp.power(pos_grid, 2), axis=-1))
            self.kernel = jnp.exp(-r_grid)

        elif kernel_type == "gaussian":
            x, y, z = jnp.meshgrid(
                jnp.arange(-kernel_side, kernel_side + 1),
                jnp.arange(-kernel_side, kernel_side + 1),
                jnp.arange(-kernel_side, kernel_side + 1),
            )
            pos_grid = jnp.stack((x, y, z), axis=-1) * cell_size
            r_grid = jnp.sqrt(jnp.sum(jnp.power(pos_grid, 2), axis=-1))
            self.kernel = jnp.exp(-(r_grid**2))

        elif kernel_type == "const":
            self.kernel = jnp.ones((kernel_size, kernel_size, kernel_size))

        elif kernel_type == "linear":
            x, y, z = jnp.meshgrid(
                jnp.arange(-kernel_side, kernel_side + 1),
                jnp.arange(-kernel_side, kernel_side + 1),
                jnp.arange(-kernel_side, kernel_side + 1),
            )
            pos_grid = jnp.stack((x, y, z), axis=-1) * cell_size
            r_grid = jnp.sqrt(jnp.sum(jnp.power(pos_grid, 2), axis=-1))

            self.kernel = jnp.min(r_grid) - r_grid

        self.kernel = self.kernel / jnp.sum(self.kernel)

    def __call__(
        self, x: Float[Array, "grid_size grid_size grid_size channel_size"]
    ) -> Float[
        Array, "pooled_grid_size pooled_grid_size pooled_grid_size channel_size"
    ]:
        """Pool the input grid using the defined kernel and stride.

        :param x: Input array representing the 3D grid with dimensions (grid_size, grid_size, grid_size, channel_size).
        :type x: jax.numpy.array
        :returns: Pooled 3D grid with reduced dimensions.
        :rtype: jax.numpy.array
        """
        # Apply the kernel
        x = jax.lax.conv_general_dilated(
            lhs=jnp.expand_dims(x, axis=0),
            rhs=jnp.tile(self.kernel, (x.shape[-1], x.shape[-1], 1, 1, 1)),
            window_strides=[1, 1, 1],
            padding="VALID",
            dimension_numbers=("NHWDC", "OIHWD", "NHWDC"),
        )[0]

        # Define a helper function to get a single reduced grid
        def _get_reduced_grid(start_i, start_j, start_k):
            return x[
                start_i :: self.stride,
                start_j :: self.stride,
                start_k :: self.stride,
                :,
            ]

        # Create all possible grids
        all_grids = []
        for i in range(self.stride):
            for j in range(self.stride):
                for k in range(self.stride):
                    all_grids.append(_get_reduced_grid(i, j, k))
        all_grids = jnp.array(all_grids)

        # Compute the norm for each possible grid
        norms = jnp.mean(jnp.power(all_grids, 2.0), axis=(1, 2, 3, 4))
        argmax = jnp.argmax(norms)

        return all_grids[argmax]


# Define the layer that compress the information in the grid in an invariant way
class compress_3d_e3(eqx.Module):
    """Compress a 3D grid with a result invariant to E(3)."""

    convs: list
    lins: list
    irreps_in: list
    irreps_out: list
    pool: list
    pad_size: list
    pad_pooling_size: list

    def __init__(
        self,
        key: Key,
        kernel_size: int = 3,
        cell_size: float = 1.0,
        conv_irreps: list = [
            "1x0e",
            "5x0e",
            "10x0e+1x2e",
            "20x0e+1x1o+2x2e",
            "64x0e",
        ],
        n_neurons_lins: list = [64, 32, 16, 8],
        n_neurons_radial: list = [4, 4],
        pooling_stride: int = 2,
        kernel_pooling_size: int = 3,
    ):
        """Initialize the class.

        :param key: Key for random number generation.
        :type key: Key
        :param kernel_size: Size of the convolutional kernels.
        :type kernel_size: int
        :param cell_size: Size of the cell in the input grid.
        :type cell_size: float
        :param conv_irreps: Irreps for the convolutional layers.
        :type conv_irreps: list
        :param n_neurons_lins: Number of neurons for the linear layers.
        :type n_neurons_lins: list
        :param n_neurons_radial: Number of neurons for the radial part.
        :type n_neurons_radial: list
        :param pooling_stride: Stride of the pooling layer.
        :type pooling_stride: int
        :param kernel_pooling_size: Size of the pooling kernel.
        :type kernel_pooling_size: int
        :raises ValueError: If `kernel_size` is an even number.
        :raises ValueError: If `kernel_pooling_size` is an even number when greater than 1.
        """
        # Check if the kernel size is odd
        if kernel_size % 2 == 0:
            raise ValueError("The kernel_size must be odd!")

        # Set the keys and other parameters
        key_conv, key_lin = jrandom.split(key, 2)
        Nconv = len(conv_irreps)
        Nlin = len(n_neurons_lins)
        keys_conv = jrandom.split(key_conv, Nconv - 1)
        keys_lin = jrandom.split(key_lin, Nlin - 1)

        # Correct the irreps to take into account the gates
        self.irreps_in = []
        self.irreps_out = []
        for i in range(Nconv - 1):
            self.irreps_in.append(e3nn.Irreps(conv_irreps[i]))
            self.irreps_out.append(e3nn.Irreps(conv_irreps[i + 1]))
            n_gates = jnp.sum(jnp.array(self.irreps_out[i].ls) > 0)
            self.irreps_out[i] = e3nn.Irreps(f"{n_gates}x0e") + self.irreps_out[i]

        # Check the number of neurons between the convolutions and the linear layers
        if self.irreps_out[-1].dim != n_neurons_lins[0]:
            raise ValueError(
                "The number of neurons between the convolutions and the linear layers must be the same!"
            )

        # Check if the last convolutions gives scalar irreps
        if self.irreps_out[-1].lmax > 0:
            raise ValueError("The last convolution must have only scalar irreps!")

        # Compute the pad sizes for the periodic boundary conditions
        kernel_side = (kernel_size - 1) // 2
        self.pad_size = [
            (kernel_side, kernel_side),
            (kernel_side, kernel_side),
            (kernel_side, kernel_side),
            (0, 0),
        ]

        # Construct the convolutional layers
        self.convs = []
        for i in range(Nconv - 1):
            self.convs.append(
                conv_e3_layer(
                    irreps_in=self.irreps_in[i],
                    irreps_out=self.irreps_out[i],
                    kernel_size=kernel_size,
                    stride=1,
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
        if kernel_pooling_size > 1:
            # Check if the kernel size is odd
            if kernel_pooling_size % 2 == 0:
                raise ValueError("The kernel_pooling_size must be odd!")

            # Compute the pad sizes for the periodic boundary conditions
            kernel_side = (kernel_pooling_size - 1) // 2
            self.pad_pooling_size = [
                (kernel_side, kernel_side),
                (kernel_side, kernel_side),
                (kernel_side, kernel_side),
                (0, 0),
            ]

            # Set the pooling layer
            self.pool = eqx.nn.AvgPool3d(
                kernel_size=kernel_pooling_size,
                stride=pooling_stride,
                padding=0,
            )
        else:
            # No padding for the pooling layer
            self.pad_pooling_size = (
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0),
            )

            # Trivial pooling
            self.pool = lambda x: x

    def compute_kernels(self):
        """Compute the kernels of all convolutional layers."""
        for conv in self.convs:
            conv.compute_kernel()

    def __call__(self, x: Float[Array, "grid_size grid_size grid_size channel_size"]):
        """Compress the given 3D grid using the equivariant convolutional layers.

        :param x: Input 3D grids.
        :type x: jax.numpy.array
        :returns: Compressed 3D grid.
        :rtype: jax.numpy.array
        """
        # Apply the conv layers
        for i, conv in enumerate(self.convs):
            x = jnp.pad(x, pad_width=self.pad_size, mode="wrap")
            x = conv(x)
            x = e3nn.gate(
                e3nn.IrrepsArray(irreps=self.irreps_out[i], array=x),
                even_act=jnn.tanh,
                even_gate_act=jnn.tanh,
                normalize_act=True,
            ).array
            x = jnp.pad(x, pad_width=self.pad_pooling_size, mode="wrap")
            x = jnp.transpose(
                self.pool(jnp.transpose(x, axes=(3, 0, 1, 2))), axes=(1, 2, 3, 0)
            )

        # Flatten the data
        x = jnp.mean(x, axis=(0, 1, 2))

        # Apply the linear layers
        for i in range(len(self.lins) - 1):
            x = jnn.relu(self.lins[i](x))
        x = self.lins[-1](x)

        return x


# Define the layer that compress the information in the grid in an invariant way
class compress_fourier_3d_e3(eqx.Module):
    """Compress a 3D grid with a result invariant to E(3) in Fourier space."""

    convs: list
    lins: list
    irreps_in: list
    irreps_out: list

    def __init__(
        self,
        key: Key,
        grid_size: list,
        cell_size: float = 1.0,
        conv_irreps: list = [
            "1x0e",
            "5x0e",
            "10x0e+1x2e",
            "20x0e+1x1o+2x2e",
            "64x0e",
        ],
        n_neurons_lins: list = [64, 32, 16, 8],
        n_neurons_radial: list = [4, 4],
        downsampling_factor: int = 1,
    ):
        """Initialize the class.

        :param key: Key for random number generation.
        :type key: Key
        :param grid_size: Dimensions of the input grid (e.g., [Nx, Ny, Nz]).
        :type grid_size: list
        :param cell_size: Size of the cell in the input grid.
        :type cell_size: float
        :param conv_irreps: Irreps for the convolutional layers.
        :type conv_irreps: list
        :param n_neurons_lins: Number of neurons for the linear layers.
        :type n_neurons_lins: list
        :param n_neurons_radial: Number of neurons for the radial part.
        :type n_neurons_radial: list
        :param downsampling_factor: Factor by which the grid size is downsampled after each convolution.
        :type downsampling_factor: int
        """
        # Set the keys
        key_conv, key_lin = jrandom.split(key, 2)
        Nconv = len(conv_irreps)
        Nlin = len(n_neurons_lins)
        keys_conv = jrandom.split(key_conv, Nconv - 1)
        keys_lin = jrandom.split(key_lin, Nlin - 1)

        # Correct the irreps to take into account the gates
        self.irreps_in = []
        self.irreps_out = []
        for i in range(Nconv - 1):
            self.irreps_in.append(e3nn.Irreps(conv_irreps[i]))
            self.irreps_out.append(e3nn.Irreps(conv_irreps[i + 1]))
            n_gates = jnp.sum(jnp.array(self.irreps_out[i].ls) > 0)
            self.irreps_out[i] = e3nn.Irreps(f"{n_gates}x0e") + self.irreps_out[i]

        # Check the number of neurons between the convolutions and the linear layers
        if self.irreps_out[-1].dim != n_neurons_lins[0]:
            raise ValueError(
                "The number of neurons between the convolutions and the linear layers must be the same!"
            )

        # Check if the last convolutions gives scalar irreps
        if self.irreps_out[-1].lmax > 0:
            raise ValueError("The last convolution must have only scalar irreps!")

        # Construct the convolutional layers
        self.convs = []
        for i in range(Nconv - 1):
            self.convs.append(
                conv_fourier_e3_layer(
                    key=keys_conv[i],
                    grid_size=grid_size,
                    irreps_in=self.irreps_in[i],
                    irreps_out=self.irreps_out[i],
                    kernel_size=int(
                        jnp.min(jnp.array(grid_size) // downsampling_factor)
                    ),
                    cell_size=cell_size,
                    n_neurons_radial=n_neurons_radial,
                )
            )
            grid_size = (
                jnp.ones(3) * jnp.min(jnp.array(grid_size) // downsampling_factor)
            ).tolist()

        # Construct the linear layers
        self.lins = []
        for i in range(Nlin - 1):
            self.lins.append(
                eqx.nn.Linear(n_neurons_lins[i], n_neurons_lins[i + 1], key=keys_lin[i])
            )

    def compute_kernels(self):
        """Compute the kernels of all convolutional layers."""
        for conv in self.convs:
            conv.compute_kernel()

    def __call__(self, x: Float[Array, "grid_size grid_size grid_size channel_size"]):
        """
        Compress the given 3D grid using the equivariant convolutional layers.

        :param x: Input 3D grids.
        :type x: jax.numpy.array
        :returns: Compressed 3D grid.
        :rtype: jax.numpy.array
        """
        # Apply the conv layers
        for i, conv in enumerate(self.convs):
            x = conv(x)
            x = e3nn.gate(
                e3nn.IrrepsArray(irreps=self.irreps_out[i], array=x),
                even_act=jnn.tanh,
                even_gate_act=jnn.tanh,
                normalize_act=True,
            ).array

        # Flatten the data
        x = jnp.mean(x, axis=(0, 1, 2))

        # Apply the linear layers
        for i in range(len(self.lins) - 1):
            x = jnn.relu(self.lins[i](x))
        x = self.lins[-1](x)

        return x


# Define the layer that compress the information in the grid with dimension n
class compress_nd(eqx.Module):
    """Compress a (1,2,3)D grid."""

    convs: list
    lins: list
    pool: eqx.nn.Pool
    pad_size: list
    pad_pooling_size: list

    def __init__(
        self,
        key: Key,
        dimension: int = 3,
        kernel_size: int = 3,
        conv_channels: list = [1, 4, 16, 32, 64],
        n_neurons_lins: list = [64, 32, 16, 8],
        pooling_stride: int = 2,
        kernel_pooling_size: int = 3,
    ):
        """
        Initialize the class.

        :param dimension: The dimension of the grid to be compressed.
        :type dimension: int
        :param key: Key for random number generation.
        :type key: Key
        :param kernel_size: Size of the convolutional kernels.
        :type kernel_size: int
        :param conv_channels: Number of channels for the convolutional layers.
        :type conv_channels: list
        :param n_neurons_lins: Number of neurons for the linear layers.
        :type n_neurons_lins: list
        :param pooling_stride: Stride of the pooling layer.
        :type pooling_stride: int
        :param kernel_pooling_size: Size of the pooling kernel.
        :type kernel_pooling_size: int
        :raises ValueError: If `kernel_pooling_size` is an even number when greater than 1.
        """
        # Check the number of neurons between the convolutions and the linear layers
        if conv_channels[-1] != n_neurons_lins[0]:
            raise ValueError(
                "The number of neurons between the convolutions and the linear layers must be the same!"
            )

        # Set the keys
        key_conv, key_lin = jrandom.split(key, 2)
        Nconv = len(conv_channels)
        Nlin = len(n_neurons_lins)
        keys_conv = jrandom.split(key_conv, Nconv - 1)
        keys_lin = jrandom.split(key_lin, Nlin - 1)

        # Compute the pad sizes for the periodic boundary conditions
        kernel_side = (kernel_size - 1) // 2
        self.pad_size = [(0, 0)]
        for i in range(dimension):
            self.pad_size.append((kernel_side, kernel_side))

        # Construct the convolutional layers
        self.convs = []
        for i in range(Nconv - 1):
            self.convs.append(
                eqx.nn.Conv(
                    num_spatial_dims=dimension,
                    in_channels=conv_channels[i],
                    out_channels=conv_channels[i + 1],
                    kernel_size=kernel_size,
                    stride=1,
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
        if kernel_pooling_size > 1:
            # Check if the kernel size is odd
            if kernel_pooling_size % 2 == 0:
                raise ValueError("The kernel_pooling_size must be odd!")

            # Compute the pad sizes for the periodic boundary conditions
            kernel_side = (kernel_pooling_size - 1) // 2
            self.pad_pooling_size = [(0, 0)]
            for i in range(dimension):
                self.pad_pooling_size.append((kernel_side, kernel_side))

            # Set the pooling layer
            if dimension == 1:
                self.pool = eqx.nn.AvgPool1d(
                    kernel_size=kernel_pooling_size,
                    stride=pooling_stride,
                    padding=0,
                )
            elif dimension == 2:
                self.pool = eqx.nn.AvgPool2d(
                    kernel_size=kernel_pooling_size,
                    stride=pooling_stride,
                    padding=0,
                )
            elif dimension == 3:
                self.pool = eqx.nn.AvgPool3d(
                    kernel_size=kernel_pooling_size,
                    stride=pooling_stride,
                    padding=0,
                )

        else:
            # No padding for the pooling layer
            self.pad_pooling_size = [(0, 0)]
            for i in range(dimension):
                self.pad_pooling_size.append((0, 0))

            # Trivial pooling
            self.pool = lambda x: x

    def compute_kernels(self):
        """
        This method is defined to make the class compatible with the other compression layers.
        """
        pass

    def __call__(self, x: Float[Array, "grid_size grid_size grid_size channel_size"]):
        """Compress the given ND grid using the convolutional layers.

        :param x: Input ND grid.
        :type x: jax.numpy.array
        :returns: Compressed 3D grid.
        :rtype: jax.numpy.array
        """
        # Transpose the input array to have the channels as the first dimension
        x = jnp.transpose(x, axes=(3, 0, 1, 2))

        # Apply the conv layers
        for conv in self.convs:
            x = jnp.pad(x, pad_width=self.pad_size, mode="wrap")
            x = conv(x)
            x = jnn.tanh(x)
            x = jnp.pad(x, pad_width=self.pad_pooling_size, mode="wrap")
            x = self.pool(x)

        # Flatten the data
        x = jnp.mean(x, axis=(1, 2, 3))

        # Apply the linear layers
        for i in range(len(self.lins) - 1):
            x = jnn.relu(self.lins[i](x))
        x = self.lins[-1](x)

        return x


# Define a compression class that does nothing
class no_compression(eqx.Module):
    """
    This class implements a compression layer that does nothing.
    """

    def __init__(self):
        """
        Initialize the class.
        """
        pass

    def compute_kernels(self):
        """
        This method is defined to make the class compatible with the other compression layers.
        """
        pass

    def __call__(self, x: Float[Array, "grid_size grid_size grid_size channel_size"]):
        """
        Compress the given array using the linear layers.
        """
        return jnp.array([])


# Define the layer that compress the information in an array without convolutions
class compress_array(eqx.Module):
    """Compress an array (don't use convolutions)."""

    lins: list

    def __init__(
        self,
        key: Key,
        n_neurons_lins: list = [32, 16, 16, 8],
    ):
        """Initialize the class.

        :param key: Key for random number generation.
        :type key: Key
        :param n_neurons_lins: Number of neurons for the linear layers.
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

    def __call__(self, x: Float[Array, "array_size"]):
        """Compress the given array using the linear layers.

        :param x: Input array.
        :type x: jax.numpy.array
        :returns: Compressed array.
        :rtype: jax.numpy.array
        """
        # Apply the linear layers
        for i in range(len(self.lins) - 1):
            x = jnn.relu(self.lins[i](x))
        x = self.lins[-1](x)

        return x


# Define the layer that concatenate the conditionals (random choice from FFJORD)
class concat_layer(eqx.Module):
    """Concatenate the parameters with time and any other conditionals."""

    concat_layer: eqx.nn.Linear
    time_dilatation: eqx.nn.Linear
    time_shift: eqx.nn.Linear

    def __init__(
        self,
        key: Key,
        in_size: int,
        out_size: int,
        compressed_grid_size: int = 0,
        compressed_array_size: int = 0,
    ):
        """Initialize the class.

        :param key: Key for random number generation.
        :type key: Key
        :param in_size: Size of the input array.
        :type in_size: int
        :param out_size: Size of the output array.
        :type out_size: int
        :param compressed_grid_size: Size of the compressed grid array.
        :type compressed_grid_size: int
        :param compressed_array_size: Size of the compressed general array.
        :type compressed_array_size: int
        """

        # Set the keys
        key_concat, key_dilat, key_shift = jrandom.split(key, 3)

        # Define the layer that concatenate everything
        self.concat_layer = eqx.nn.Linear(
            in_size + compressed_grid_size + compressed_array_size,
            out_size,
            key=key_concat,
        )

        # Define the layers that transform the time coordinate
        self.time_dilatation = eqx.nn.Linear(1, out_size, key=key_dilat)
        self.time_shift = eqx.nn.Linear(1, out_size, use_bias=False, key=key_shift)

    def __call__(
        self,
        t: float,
        theta: Float[Array, "in_size"],
        compressed_grid: Float[Array, "compressed_grid_size"] = jax.numpy.array([]),
        compressed_array: Float[Array, "compressed_array_size"] = jax.numpy.array([]),
    ) -> Float[Array, "out_size"]:
        """Apply the layer to concatenate the input array with the time and the compressed conditionals.

        :param t: Time in the ODE.
        :type t: float
        :param y: Array in the ODE.
        :type y: jax.numpy.array
        :param compressed_grid: Compressed conditional input grid array.
        :type compressed_grid: jax.numpy.array
        :param compressed_array: Compressed conditional general array.
        :type compressed_array: jax.numpy.array
        :returns: The input array concatenated with the time and conditional.
        :rtype: jax.numpy.array
        """
        # Transform t to an array
        t_array = jnp.asarray(t)[None]

        # Compute the concatenation
        y_stacked = jnp.hstack([theta, compressed_grid, compressed_array])
        y = self.concat_layer(y_stacked) * jnn.sigmoid(
            self.time_dilatation(t_array)
        ) + self.time_shift(t_array)

        return y

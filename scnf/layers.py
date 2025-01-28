"""
This module implements some layers used in the construction of different types of continuous normalizing flows.
"""

# Import general libreries
from typing import List

import e3nn_jax as e3nn
import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom

# Import jax based libraries
from jaxtyping import Array, Float, List, PRNGKeyArray


# Class with one equivariant under E(3) layer
class E3conv_layer(eqx.Module):
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

        :param irreps_in:
        :type irreps_in: e3nn_jax.Irreps
        :param irreps_out:
        :type irreps_out: e3nn_jax.Irreps
        :param key:
        :type key: jax.random.PRNGKeyArray
        :param kernel_size:
        :type kernel_size: int
        :param stride:
        :type stride: int
        :param cell_size:
        :type cell_size: float
        :param Nneurons_radial:
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
        j_min = abs(irreps_in.lmax - irreps_out.lmax)
        j_max = irreps_in.lmax + irreps_out.lmax

        # If the kernel is too small sets a smaller Jmax
        if j_max > 3 * kernel_side**2:
            j_max = int(3 * kernel_side**2)

        # Compute the spherical harmonics used
        yj = []
        for i in range(j_min, j_max + 1):
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
                    if J > Jmax:
                        continue

                    # Compute the Clebsch Gordan coefficients
                    cg = jnp.array(
                        e3nn.clebsch_gordan(int(jin), int(jout), int(J))
                    ) * jnp.sqrt(2.0 * J + 1.0)

                    # Compute the outer product
                    ttmpk.append(
                        jnp.einsum(
                            "jilmn,lmn->jilmn",
                            jnp.einsum("ijk,lmnk->jilmn", cg, YJ[J]),
                            WJ[J],
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
        # Check the inputs
        if x.irreps != self.irreps_in:
            raise ValueError(
                "The irreps of the input is incompatile with the one used to initialize the class!"
            )

        # Take the input array
        y = jnp.transpose(x.array, (3, 0, 1, 2))

        # Compute the convolution of the kernel with the input array
        y = jax.lax.conv_general_dilated(
            lhs=jnp.expand_dims(y, axis=0),
            rhs=self.Kernel[0],
            window_strides=[self.stride, self.stride, self.stride],
            padding="VALID",
        )
        return e3nn.IrrepsArray(self.irreps_out, jnp.transpose(y[0], (1, 2, 3, 0)))

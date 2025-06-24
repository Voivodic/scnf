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
import layers
# from . import layers


# Define the vector field used into the neural ordinary differential equation
class vector_field(eqx.Module):
    """
    This class implements the vector field for the continuous normalizing flow,
    which is a neural ordinary differential equation (NODE).

    It processes input data (theta) by concatenating it with compressed
    grid and array data through a series of layers.
    """

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
        grid_size: Union[list, None] = None,
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
                    key=keys_concat[
                        i
                    ],  # Corrected to use individual key for each layer
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
        """
        Computes the kernels for all convolutional layers within the vector field.
        This method iterates through the `convs` attribute (though it's not
        explicitly defined in the class, assuming it refers to internal
        convolutional components within compression layers if they exist).
        """
        for compression in (
            self.compression_grid_layers
        ):  # Assuming 'self.convs' exists within compression layers
            compression.compute_kernels()

    # Compress the array
    def compress_array(self, array: Float[Array, "batch_size array_size"]):
        """
        Compresses the input array data using the initialized array compression layers.
        The compressed data is stored internally in `self.compressed_array_data`.

        :param array: The input array to be compressed.
        :type array: Float[Array, "array_size"]
        """
        for i, compress_array_layer in enumerate(self.compression_array_layers):
            if compress_array_layer is not None:
                self.compressed_array_data[i] = compress_array_layer(array)

    # Compress the grid
    def compress_grid(
        self,
        grid: Float[Array, "batch_size grid_size grid_size grid_size channel_size"],
    ):
        """
        Compresses the input grid data using the initialized grid compression layers.
        The compressed data is stored internally in `self.compressed_grid_data`.

        :param grid: The input grid to be compressed, typically 3D with channels.
        :type grid: Float[Array, "grid_size grid_size grid_size channel_size"]
        """
        for i, compress_grid_layer in enumerate(self.compression_grid_layers):
            if compress_grid_layer is not None:
                self.compressed_grid_data[i] = jax.vmap(compress_grid_layer)(grid)
    # Reshape the compressed grid and array data
    def reshape_compressed_data(self, batch_size: int):
        """
        Reshapes the internally stored compressed grid and array data.

        This method is used to reshape the `compressed_grid_data` and
        `compressed_array_data` lists. If any of the internal arrays
        have an initial dimension of 0 (indicating they are empty placeholders),
        they are reshaped to `[batch_size, -1]`, preparing them for batch processing.

        :param batch_size: The desired batch size to reshape the data to.
        :type batch_size: int
        """
        for i in range(len(self.compressed_grid_data)):
            if self.compressed_grid_data[i].shape[0] == 0:
                self.compressed_grid_data[i] = self.compressed_grid_data[i].reshape(
                    [batch_size, -1]
                )

        for i in range(len(self.compressed_array_data)):
            if self.compressed_array_data[i].shape[0] == 0:
                self.compressed_array_data[i] = self.compressed_array_data[i].reshape(
                    [batch_size, -1]
                )

    # Compute the vector field for a given vector and time
    def __call__(
        self, t: float, theta: Float[Array, "in_size"], args=None
    ) -> Float[Array, "in_size"]:
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
        for i, concat_layer in enumerate(self.concatenation_layers):
            theta = jax.vmap(concat_layer, in_axes=(None, 0, 0, 0))(
                t, theta, self.compressed_grid_data[i], self.compressed_array_data[i]
            )
            theta = jnn.tanh(theta)

        return theta


# Define the class for the continuous normalizing flow
class cnf(eqx.Module):
    """
    Continuous Normalizing Flow (CNF) class that implements a neural ODE-based flow.
    """

    vector_fields: list  # List of vector field transformations
    y_size: int          # Dimension of the target space
    x_size: int          # Dimension of the conditional input (0 if unconditional)
    t0: float            # Initial time
    t1: float            # Final time
    dt0: float           # Initial step size
    mean_std_layer: Union[eqx.Module, None]  # Module for computing mean/std of base distribution
    compact_type: str    # Type of compression used ('standard' or 'equivariant')

    def __init__(
        self,
        key: PRNGKeyArray,
        n_neurons: list,
        n_blocks: int = 1,
        n_neurons_mean_std: list = [],
        grid_size: Union[list, None] = None,
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
        # Save parameters
        self.y_size = y_size
        self.x_size = x_size
        self.t0 = 0.0
        self.t1 = 1.0
        self.dt0 = 0.1

        # Split the key
        keys = jrandom.split(key, n_blocks+1)
        
        # Initialize vector fields
        self.vector_fields = []
        for k in keys[:n_blocks]:
            self.vector_fields.append(
                vector_field(
                    key=k,
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
        # Use only one vector field for it
        if len(n_neurons_mean_std) > 0:
            self.mean_std_layers = vector_field(
                key=keys[-1],
                n_neurons=n_neurons_mean_std,
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
        else:
            self.mean_std_layer = None

    def log_normal(self, y: Float[Array, "y_size"], x: Float[Array, "x_size"] = None):
        """
        Compute log probability under normal distribution.

        :param y: Input samples.
        :type y: Float[Array, "y_size"]
        :param x: Conditional input (optional).
        :type x: Float[Array, "x_size"], optional
        :return: Log probability of y under the base distribution.
        :rtype: float
        """
        if x is None:
            return -0.5 * (y.shape[-1] * jnp.log(2 * jnp.pi) + jnp.sum(y**2))
        else:
            # Compute mean and std from conditional input
            if self.mean_std_layers is not None:
                self.mean_std_layers.compress_grid(x)
                self.mean_std_layers.reshape_compressed_data(x.shape[0])
                mu_std = jax.vmap(self.mean_std_layers, in_axes=(None, 0, None))(
                    0.0, jnp.zeros(self.y_size), None
                )
                mu, std = jnp.split(mu_std, 2, axis=-1)
                std = jnp.exp(std)
            else:
                mu, std = 0.0, 1.0

            return -0.5 * (
                y.shape[-1] * jnp.log(2 * jnp.pi)
                + 2.0 * jnp.sum(jnp.log(std), axis=-1)
                + jnp.sum((y - mu) ** 2 / std**2, axis=-1)
            )

    def _wrapper_func_trjac_approx(
        self, t: Float, y_trjac: tuple, args: tuple
    ) -> tuple:
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
        y, trjac = y_trjac
        eps, vector_field = args

        # Compute vector field
        fn = lambda y: vector_field(t, y, ())
        f, vjp_fn = jax.vjp(fn, y)
        (eps_dfdy,) = vjp_fn(eps)
        trjac += jnp.sum(eps_dfdy * eps)

        return f, trjac

    def get_logP(
        self,
        y: Float[Array, "y_size"],
        key: PRNGKeyArray,
        save_ts: Float[Array, "Ntimes"],
        x: Float[Array, "x_size"] = None,
    ) -> tuple:
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
        saveat = df.SaveAt(ts=save_ts)
        term = df.ODETerm(self._wrapper_func_trjac_approx)
        solver = df.Tsit5()
        eps = jrandom.normal(key, y.shape)

        # Solve backward in time for all blocks
        delta_log_likelihood = 0.0
        transformed = []
        for vf in reversed(self.vector_fields):
            # Prepare conditional input
            if x is not None:
                vf.compress_grid(x)
                vf.reshape_compressed_data(x.shape[0])
                transformed.append(vf.compressed_grid_data[0])

            # Solve ODE
            y = (y, delta_log_likelihood)
            sol = df.diffeqsolve(
                term,
                solver,
                self.t1,
                self.t0,
                -self.dt0,
                y,
                (eps, vf),
                saveat=saveat,
                stepsize_controller=df.PIDController(rtol=1e-4, atol=1e-4),
            )
            y, delta_log_likelihood = sol.ys

        return y, delta_log_likelihood[-1] + self.log_normal(y[-1, :], x), jnp.array(transformed)

    @eqx.filter_jit
    def solve_ODE(
        self, y: Float[Array, "y_size"], Ntimes: int = 1
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
        # Set up time points
        save_ts = jnp.linspace(self.t0, self.t1, Ntimes - 1, endpoint=False)
        save_ts = jnp.hstack([save_ts, self.t1])
        saveat = df.SaveAt(ts=save_ts)

        # Solve ODE for each vector field
        solver = df.Tsit5()
        out = []
        for vf in self.vector_fields:
            sol = df.diffeqsolve(
                df.ODETerm(vf),
                solver,
                self.t0,
                self.t1,
                self.dt0,
                y,
                saveat=saveat,
                stepsize_controller=df.PIDController(rtol=1e-5, atol=1e-5),
            )
            out.append(sol.ys)
            y = sol.ys[-1]

        return jnp.concatenate(out)

    def sample(
        self,
        key: PRNGKeyArray,
        Nsamples: int = 1000,
        x: Float[Array, "x_size"] = None,
        prior: Callable = lambda _: 1,
        max_prior: float = 0.0,
        Nmax: int = 10_000,
        Ntimes: int = 1,
    ) -> Float[Array, "Nsamples ..."]:
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
        # Compute kernels if needed
        if self.compact_type == "equivariant":
            for vf in self.vector_fields:
                vf.compute_kernels()
            if self.mean_std_layers is not None:
                self.mean_std_layers.compute_kernels()

        # Set up conditional input
        if x is not None:
            for vf in self.vector_fields:
                vf.compress_grid(x)
                vf.reshape_compressed_data(x.shape[0])
            if self.mean_std_layers is not None:
                mu, std = self.mean_std(x)
            else:
                mu, std = 0.0, 1.0
        else:
            mu, std = 0.0, 1.0

        # Generate samples
        Nout = 0
        count = 0
        y_out = jnp.empty([0, Ntimes * len(self.vector_fields), self.y_size])
        
        while Nout < Nsamples and count < Nmax:
            key, key_normal, key_uni = jrandom.split(key, 3)
            
            # Generate initial samples
            y_ini = jrandom.normal(key_normal, (Nsamples, self.y_size)) * std + mu
            
            # Evolve ODE
            y = jax.vmap(self.solve_ODE, in_axes=(0, None))(y_ini, Ntimes)
            
            # Apply prior and rejection sampling
            if max_prior != 0.0:
                P_prior = jax.vmap(prior)(y[:, -1, :]) / max_prior
            else:
                P_prior = jax.vmap(prior)(y[:, -1, :])
                P_prior = P_prior / jnp.max(P_prior)
            
            eps = jrandom.uniform(key_uni, shape=(Nsamples,))
            mask = eps < P_prior
            y_out = jnp.vstack([y_out, y[mask, :]])
            Nout = y_out.shape[0]
            count += 1

        return y_out[:Nsamples, :] if Ntimes > 1 else y_out[:Nsamples, 0, :]

    def __getstate__(self):
        """Get state for serialization."""
        return self.__dict__

    def __setstate__(self, state):
        """Set state from serialization."""
        self.__dict__.update(state)

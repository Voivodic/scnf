"""
This module implements the main class to run the inference of the models.
"""

# Import the main libraries
import equinox as eqx
import diffrax as df
import optax
from tqdm import tqdm
import os
import h5py as h5

# Import the jax related modules
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from jaxtyping import Array, Float, PRNGKeyArray, Union
from typing import Callable
from optax import tree_utils as otu

# Import the modules used
from . import cnf as cnf_module


# Define the loss function used in the training
def loss(
    diff_model: cnf_module.cnf,
    static_model: cnf_module.cnf,
    key: PRNGKeyArray,
    theta: Float[Array, "theta_size"],
    grid: Float[Array, "grid_size grid_size grid_size channel_size"] = jnp.array([]),
    array: Float[Array, "array_size"] = jnp.array([]),
):
    """
    Compute the loss for the CNF model.

    :param diff_model: Differentiable part of the model.
    :type diff_model: cnf
    :param static_model: Static part of the model.
    :type static_model: cnf
    :param key: Random key.
    :type key: PRNGKeyArray
    :param theta: The parameters to be fitted.
    :type theta: jax.numpy.array
    :param grid: The conditional grid.
    :type grid: jax.numpy.array
    :param array: The conditional array.
    :type array: jax.numpy.array
    :return: Computed loss value.
    :rtype: float
    """
    # Combine the differentiable and the static parts of the model
    model = eqx.combine(diff_model, static_model)

    # Pre-compute the kernels used for the convolution in each layer
    model.compute_kernels()

    # Solve backward the ODE to get the logP
    _, logP = jax.vmap(model.get_logP, in_axes=(None, 0, 0, 0))(
        key, theta, grid, array
    )

    return -jnp.mean(logP)


# Function that updates the weights one step
@eqx.filter_jit
def make_step(
    loss_func: Callable,
    model: cnf_module.cnf,
    optim: optax.GradientTransformation,
    key: PRNGKeyArray,
    theta: Float[Array, "theta_size"],
    grid: Float[Array, "grid_size grid_size grid_size channel_size"],
    array: Float[Array, "array_size"],
    optim_state: tuple,
    lr_schedule_state: optax.OptState,
):
    """
    Perform one training step.

    :param loss_func: Loss function.
    :type loss_func: Callable
    :param model: CNF model.
    :type model: cnf
    :param optim: Optimizer.
    :type optim: optax.GradientTransformation
    :param key: Random key.
    :type key: PRNGKeyArray
    :param theta: The parameters to be fitted.
    :type theta: jax.numpy.array
    :param grid: The conditional grid.
    :type grid: jax.numpy.array
    :param array: The conditional array.
    :type array: jax.numpy.array
    :param optim_state: Optimizer state.
    :type optim_state: tuple
    :param lr_schedule_state: Learning rate schedule state.
    :type lr_schedule_state: optax.OptState
    :return: Updated model, loss value, and optimizer state.
    :rtype: tuple
    """

    # Split the model between the trainable and fixed parameters
    diff_model, static_model = eqx.partition(
        model, lambda x: isinstance(x, eqx.nn.Linear)
    )

    # Compute the loss and it gradient
    loss_value, grads = eqx.filter_value_and_grad(loss_func)(
        diff_model, static_model, key, theta, grid, array
    )

    # Update the state and the learning rate
    updates, optim_state = optim.update(grads, optim_state, diff_model)
    updates = otu.tree_scalar_mul(lr_schedule_state.scale, updates)

    # Apply the updates
    diff_model = eqx.apply_updates(diff_model, updates)

    # Combine the differentiable and the static parts of the model
    model = eqx.combine(diff_model, static_model)

    return model, loss_value, optim_state


# Define the class that run the inference
class Inference(eqx.Module):
    models: list
    n_train: int
    n_validation: int
    loss_best: list
    losses: Float[Array, "Nsteps"]
    losses_validation: Float[Array, "Nsteps"]
    data_loader_theta: Callable
    data_loader_grid: Callable
    data_loader_array: Callable
    lr_history: list

    # Initialize the class used for the inference
    def __init__(
        self,
        key: PRNGKeyArray,
        data: dict,
        n_validation: int = 0,
        model: cnf_module.cnf = None,
        n_neurons: list = [64, 64, 64],
        grid_size: list = [],
        kernel_size: int = 3,
        cell_size: float = 1.0,
        conv_irreps: list = [],
        n_neurons_radial: list = [4],
        n_neurons_lins: list = [],
        n_neurons_array: list = [],
        pooling_stride: int = 2,
        kernel_pooling_size: int = 3,
        n_channels: list = [],
        conv_space: str = "configuration",
        t0: float = 0.0,
        t1: float = 1.0,
        dt0: float = 0.1,
    ):
        """
        Initialize the inference class.

        :param key: Random key for initialization.
        :type key: PRNGKeyArray
        :param data: Dictionary with the data information.
        :type data: dict
        :param n_validation: Number of validation data.
        :type n_validation: int, optional
        :param model: CNF model.
        :type model: cnf_module.cnf, optional
        :param n_neurons: List of neurons for each layer in the vector field.
        :type n_neurons: list, optional
        :param grid_size: Size of the input grid data.
        :type grid_size: list, optional
        :param kernel_size: Size of convolutional kernels.
        :type kernel_size: int, optional
        :param cell_size: Size of the simulation cell for E3 convolutions.
        :type cell_size: float, optional
        :param conv_irreps: Irreps for E3 convolutions.
        :type conv_irreps: list, optional
        :param n_neurons_radial: Neurons in radial layers for E3 convolutions.
        :type n_neurons_radial: list, optional
        :param n_neurons_lins: Neurons in linear layers for grid compression.
        :type n_neurons_lins: list, optional
        :param n_neurons_array: Neurons in linear layers for array compression.
        :type n_neurons_array: list, optional
        :param pooling_stride: Stride for pooling operations.
        :type pooling_stride: int, optional
        :param kernel_pooling_size: Kernel size for pooling.
        :type kernel_pooling_size: int, optional
        :param n_channels: Channel sizes for ND convolutions.
        :type n_channels: list, optional
        :param conv_space: Space for convolutions ('configuration' or 'fourier').
        :type conv_space: str, optional
        :param t0: Initial time of the ODE.
        :type t0: float, optional
        :param t1: Final time of the ODE.
        :type t1: float, optional
        :param dt0: Initial step size of the ODE.
        :type dt0: float, optional
        """
        # Check if the data dictionary has the required keys
        if not all(
            key in data
            for key in [
                "n_train",
                "data_loader_theta",
                "data_loader_grid",
                "data_loader_array",
            ]
        ):
            raise ValueError(
                "The data dictionary must contain the number of simulations (n_train), and the data loaders for theta, grid and array!"
            )

        # Set the number of training and validation data
        self.n_train = data["n_train"]
        self.n_validation = n_validation

        # Set the data loaders
        self.data_loader_theta = data["data_loader_theta"]
        self.data_loader_grid = data["data_loader_grid"]
        self.data_loader_array = data["data_loader_array"]

        # Initialize the model
        if model is None:
            self.models = [
                cnf_module.cnf(
                    key=key,
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
                    t0=t0,
                    t1=t1,
                    dt0=dt0,
                )
            ]
        else:
            self.models = [model]

        # Set the best model and losses
        self.models.append(self.models[0])
        self.models.append(self.models[0])
        self.loss_best = [jnp.inf, jnp.inf]
        self.losses = []
        self.losses_validation = []
        self.lr_history = []

    # Get the best model and loss
    def get_best(self):
        if self.n_validation > 0:
            return (
                self.models[1],
                self.loss_best[0],
                self.losses,
                self.models[2],
                self.loss_best[1],
                self.losses_validation,
            )
        else:
            return self.models[1], self.loss_best[0], self.losses

    # Compute the logP for some data (used to compute in the validation set)
    def logP_validation(self, n_batches: int, key: PRNGKeyArray):
        # Compute the batch size
        batch_size = int(self.n_validation / n_batches)

        # Define the indexes for the validation
        inds = jnp.arange(self.n_train, self.n_train + self.n_validation)

        # Pre-compute the kernels used in the convolutions
        self.models[0].compute_kernels()

        # Make one step for each batch
        logP = 0.0
        key_loss = jrandom.split(key, n_batches)
        for i in range(n_batches):
            theta = self.data_loader_theta(inds[i * batch_size : (i + 1) * batch_size])
            grid = self.data_loader_grid(inds[i * batch_size : (i + 1) * batch_size])
            array = self.data_loader_array(inds[i * batch_size : (i + 1) * batch_size])

            _, logP_batch = jax.vmap(
                self.models[0].get_logP, in_axes=(None, 0, 0, 0)
            )(key_loss[i], theta, grid, array)
            logP += jnp.mean(logP_batch)

        # Save the loss of this step
        logP = logP / n_batches

        return -logP

    # Train the CNF
    def train(
        self,
        n_steps: int,
        n_batches: int,
        print_every: int,
        optim: optax._src.base.GradientTransformationExtraArgs,
        key: PRNGKeyArray,
        optim_state: tuple = None,
        lr_schedule: optax._src.base.GradientTransformationExtraArgs = None,
        lr_schedule_state: tuple = None,
        suffix: str = None,
        lr_limit: float = 1e-4,
    ):
        # Compute the batch size
        batch_size = int(self.n_train / n_batches)

        # Create a folder for the output
        if suffix is not None:
            try:
                os.system("mkdir Outputs/")
            except:
                pass

        # Create the first optim state
        if optim_state is None:
            diff_model, _ = eqx.partition(
                self.models[0], lambda x: isinstance(x, eqx.nn.Linear)
            )
            optim_state = optim.init(diff_model)

        # Create the learning rate schedule
        if lr_schedule is None:
            lr_schedule = optax.contrib.reduce_on_plateau(
                patience=10, cooldown=0, factor=0.5, rtol=1e-5
            )

        # Create the first state of the lr_chedule
        if lr_schedule_state is None:
            diff_model, _ = eqx.partition(
                self.models[0], lambda x: isinstance(x, eqx.nn.Linear)
            )
            lr_schedule_state = lr_schedule.init(diff_model)

        # Create the key used in each step to split the data in batches
        key_step, key_validation = jrandom.split(key, 2)
        keys_steps = jrandom.split(key_step, n_steps)

        # Run the main training loop
        for step, bkey in enumerate(tqdm(keys_steps)):
            # Indexes used to shuffle the dataset
            key_shuffle, key_losses = jrandom.split(bkey, 2)
            inds = jrandom.permutation(key_shuffle, jnp.arange(self.n_train))

            # Compute the loss for the validation
            if self.n_validation > 0:
                self.losses_validation.append(
                    self.logP_validation(n_batches=n_batches, key=key_validation)
                )
                lr_loss = self.losses_validation[-1]

                # Save the best model so far in the validation
                if self.losses_validation[-1] < self.loss_best[1]:
                    self.models[2] = self.models[0]
                    self.loss_best[1] = self.losses_validation[-1]
            else:
                lr_loss = self.losses[-1] if len(self.losses) > 0 else jnp.inf

            # Make one step for each batch
            loss_step = 0.0
            key_loss = jrandom.split(key_losses, n_batches)
            for i in range(n_batches):
                # Make update the network weigts using this batch
                theta = self.data_loader_theta(
                    inds[i * batch_size : (i + 1) * batch_size]
                )
                grid = self.data_loader_grid(
                    inds[i * batch_size : (i + 1) * batch_size]
                )
                array = self.data_loader_array(
                    inds[i * batch_size : (i + 1) * batch_size]
                )

                self.models[0], loss_batch, optim_state = make_step(
                    loss,
                    self.models[0],
                    optim,
                    key_loss[i],
                    theta,
                    grid,
                    array,
                    optim_state,
                    lr_schedule_state,
                )

                # Save the loss of this batch
                loss_step += loss_batch

            # Save the loss of this step
            (self.losses).append(loss_step / n_batches)

            # Save the best model so far in the training
            if self.losses[-1] < self.loss_best[0]:
                self.models[1] = self.models[0]
                self.loss_best[0] = self.losses[-1]

            # Adjusts the learning rate scaling value
            diff_model, _ = eqx.partition(
                self.models[0], lambda x: isinstance(x, eqx.nn.Linear)
            )
            _, lr_schedule_state = lr_schedule.update(
                updates=diff_model, state=lr_schedule_state, value=lr_loss
            )
            self.lr_history.append(lr_schedule_state.scale)

            # Print partial results and save the models
            if step % print_every == 0:
                if type(suffix) == str:
                    try:
                        os.system("rm Outputs/Losses_%s.h5" % (suffix))
                    except:
                        pass

                if self.n_validation > 0:
                    print(
                        "Step = %d, Loss_training = %.4f, Loss_validation = %.4f"
                        % (step, self.losses[-1], self.losses_validation[-1])
                    )

                    if type(suffix) == str:
                        # Save models (only the parameters fitted)
                        diff_model, _ = eqx.partition(
                            self.models[1], lambda x: isinstance(x, eqx.nn.Linear)
                        )
                        eqx.tree_serialise_leaves(
                            "Outputs/Model_training_%s.eqx" % (suffix), diff_model
                        )

                        diff_model, _ = eqx.partition(
                            self.models[2], lambda x: isinstance(x, eqx.nn.Linear)
                        )
                        eqx.tree_serialise_leaves(
                            "Outputs/Model_validation_%s.eqx" % (suffix), diff_model
                        )

                        # Save losses
                        f = h5.File("Outputs/Losses_%s.h5" % (suffix), "w")
                        f.create_dataset("loss_training", data=self.losses)
                        f.create_dataset(
                            "loss_validation", data=self.losses_validation
                        )
                        f.create_dataset("lr_history", data=self.lr_history)
                        f.close()

                else:
                    print(
                        "Step = %d, Loss_training = %.4f" % (step, self.losses[-1])
                    )

                    if type(suffix) == str:
                        # Save model (only the parameters fitted)
                        diff_model, _ = eqx.partition(
                            self.models[1], lambda x: isinstance(x, eqx.nn.Linear)
                        )
                        eqx.tree_serialise_leaves(
                            "Outputs/Model_training_%s.eqx" % (suffix), diff_model
                        )

                        # Save losses
                        f = h5.File("Outputs/Losses_%s.h5" % (suffix), "w")
                        f.create_dataset("loss_training", data=self.losses)
                        f.create_dataset("lr_history", data=self.lr_history)
                        f.close()

            # Stop the loop if the learning rate got very small
            if lr_schedule_state.scale <= lr_limit:
                print("The CNF converged!")
                break

        # Return the state
        return optim_state
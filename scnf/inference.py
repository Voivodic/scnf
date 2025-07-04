"""
This module implements the main class to run the inference of the models.
"""

# Import the main libraries
import equinox as eqx
import diffrax as df
import optax
import os
import h5py as h5

# Import the jax related modules
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree as jtree
from jaxtyping import Array, Float, PRNGKeyArray, Union
from typing import Callable
from optax import tree_utils as otu

# Import the modules used
from . import cnf


# Define the loss function used in the training
def loss(
    diff_model: cnf.cnf,
    static_model: cnf.cnf,
    key: PRNGKeyArray,
    theta: Float[Array, "batch_size theta_size"],
    grid: Float[
        Array, "batch_size grid_size grid_size grid_size channel_size"
    ] = jnp.array([]),
    array: Float[Array, "batch_size array_size"] = jnp.array([]),
    saveat: df.SaveAt = df.SaveAt(ts=jnp.array([0.0])),
    poly_project: Float[Array, "n_times n_times"] = jnp.array([1.0]),
    alpha_reg: float = 0.0,
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
    theta, logP = jax.vmap(model.get_logP, in_axes=(None, 0, 0, 0, None))(
        key, theta, grid, array, saveat
    )

    # Compute the loss from the polynomial regularization
    loss_reg = jnp.mean(jnp.matmul(poly_project, theta) ** 2)

    return -jnp.mean(logP) + alpha_reg * loss_reg


# Function that updates the weights one step
@eqx.filter_jit
def make_step(
    key: PRNGKeyArray,
    model: cnf.cnf,
    model_mask: cnf.cnf,
    optim: optax.GradientTransformation,
    theta: Float[Array, "batch_size theta_size"],
    grid: Float[Array, "batch_size grid_size grid_size grid_size channel_size"],
    array: Float[Array, "batch_size array_size"],
    optim_state: tuple,
    lr_schedule_state: optax.OptState,
    saveat: df.SaveAt,
    poly_project: Float[Array, "n_times n_times"],
    alpha_reg: float,
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
    diff_model, static_model = eqx.partition(model, model_mask)

    # Compute the loss and it gradient
    loss_value, grads = eqx.filter_value_and_grad(loss)(
        diff_model,
        static_model,
        key,
        theta,
        grid,
        array,
        saveat,
        poly_project,
        alpha_reg,
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
class inference(eqx.Module):
    model: [cnf.cnf]
    model_mask: cnf.cnf
    n_train: int
    n_validation: int
    losses_best: list
    losses_train: Float[Array, "n_epochs"]
    losses_validation: Float[Array, "n_epochs"]
    data_loader_theta: Callable
    data_loader_grid: Callable
    data_loader_array: Callable
    lr_history: list
    folder_name: str

    # Initialize the class used for the inference
    def __init__(
        self,
        key: PRNGKeyArray,
        n_train: int,
        n_validation: int,
        data_loader_theta: Callable,
        data_loader_grid: Callable,
        data_loader_array: Callable,
        model: cnf.cnf = None,
        folder_name: str = "Outputs",
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
        # Set the folder name
        self.folder_name = folder_name
        os.system("mkdir -p %s" % (folder_name))

        # Set the number of training and validation data
        self.n_train = n_train
        self.n_validation = n_validation

        # Set the data loaders
        self.data_loader_theta = data_loader_theta
        self.data_loader_grid = data_loader_grid
        self.data_loader_array = data_loader_array

        # Get the model
        self.model = [model]

        # Get the mask with True in the differentiable part
        self.model_mask = cnf.get_mask(self.model[0])

        # Set the losses
        self.losses_best = [jnp.inf, jnp.inf]
        self.losses_train = []
        self.losses_validation = []
        self.lr_history = []

    # Compute the logP for some data (used to compute in the validation set)
    def logP_validation(self, batch_size: int, key: PRNGKeyArray):
        # Compute the number of batches
        n_batches = int(self.n_validation // batch_size)

        # Define the indexes for the validation
        inds = jnp.arange(self.n_train, self.n_train + self.n_validation)

        # Pre-compute the kernels used in the convolutions
        self.model[0].compute_kernels()

        # Compute the points to save at
        saveat = self.model[0].get_saveat(n_times=1, reverse=False)

        # Compute the logP for each batch
        logP = 0.0
        key_loss = jrandom.split(key, n_batches)
        for i in range(n_batches):
            theta = jax.vmap(self.data_loader_theta)(
                inds[i * batch_size : (i + 1) * batch_size]
            )
            grid = jax.vmap(self.data_loader_grid)(
                inds[i * batch_size : (i + 1) * batch_size]
            )
            array = jax.vmap(self.data_loader_array)(
                inds[i * batch_size : (i + 1) * batch_size]
            )

            _, logP_batch = jax.vmap(
                self.model[0].get_logP, in_axes=(None, 0, 0, 0, None)
            )(key_loss[i], theta, grid, array, saveat)
            logP += jnp.mean(logP_batch)

        # Save the loss of this step
        logP = logP / n_batches

        return -logP

    # Train the CNF
    def train(
        self,
        key: PRNGKeyArray,
        n_epochs: int,
        batch_size: int,
        optim: optax._src.base.GradientTransformationExtraArgs,
        print_every: int = 1,
        optim_state: Union[tuple, None] = None,
        lr_schedule: Union[
            optax._src.base.GradientTransformationExtraArgs, None
        ] = None,
        lr_schedule_state: Union[tuple, None] = None,
        suffix: str = "",
        lr_limit: float = 1e-4,
        poly_order: int = 1,
        alpha_reg: float = 0.0,
    ):
        # Check if suffix is a string
        if suffix == "":
            suffix = f"{self.n_train}_{self.n_validation}_{n_epochs}_{batch_size}"

        # Compute the number of batches
        n_batches = int(self.n_train // batch_size)

        # Get the model only with the trainable parameters
        diff_model = eqx.filter(self.model[0], self.model_mask)

        # Create the first optim state
        if optim_state is None:
            optim_state = optim.init(diff_model)

        # Create the learning rate schedule
        if lr_schedule is None:
            lr_schedule = optax.contrib.reduce_on_plateau(
                patience=10, cooldown=0, factor=0.5, rtol=1e-4
            )
            lr_loss = jnp.inf

        # Create the first state of the lr_chedule
        if lr_schedule_state is None:
            lr_schedule_state = lr_schedule.init(diff_model)

        # Compute the projection matrix for the polynomial time regularization
        n_times = int((self.model[0].t1 - self.model[0].t0) // self.model[0].dt0)
        saveat = self.model[0].get_saveat(n_times=n_times, reverse=True)
        save_ts = saveat.subs.ts
        poly_project = jnp.array([save_ts**i for i in range(poly_order + 1)]).T
        poly_project = jnp.identity(n_times) - jnp.matmul(
            jnp.matmul(
                poly_project,
                jnp.linalg.inv(jnp.matmul(jnp.transpose(poly_project), poly_project)),
            ),
            jnp.transpose(poly_project),
        )

        # Create the key used in each step to split the data in batches
        key_step, key_validation = jrandom.split(key, 2)
        keys_steps = jrandom.split(key_step, n_epochs)

        # Run the main training loop
        for step, bkey in enumerate(keys_steps):
            # Indexes used to shuffle the dataset
            key_shuffle, key_losses = jrandom.split(bkey, 2)
            inds = jrandom.permutation(key_shuffle, jnp.arange(self.n_train))

            # Compute the loss for the validation
            if self.n_validation > 0:
                self.losses_validation.append(
                    self.logP_validation(batch_size=batch_size, key=key_validation)
                )
                lr_loss = self.losses_validation[-1]

                # Save the best model so far in the validation
                if self.losses_validation[-1] < self.losses_best[1]:
                    diff_model = eqx.filter(self.model[0], self.model_mask)
                    eqx.tree_serialise_leaves(
                        "Outputs/Model_validation_%s.eqx" % (suffix), diff_model
                    )
                    self.losses_best[1] = self.losses_validation[-1]

            # Make one step for each batch
            loss_step = 0.0
            key_loss = jrandom.split(key_losses, n_batches)
            for i in range(n_batches):
                # Get the data for this batch
                theta = jax.vmap(self.data_loader_theta)(
                    inds[i * batch_size : (i + 1) * batch_size]
                )
                grid = jax.vmap(self.data_loader_grid)(
                    inds[i * batch_size : (i + 1) * batch_size]
                )
                array = jax.vmap(self.data_loader_array)(
                    inds[i * batch_size : (i + 1) * batch_size]
                )

                # Perform one step for this batch
                self.model[0], loss_batch, optim_state = make_step(
                    key=key_loss[i],
                    model=self.model[0],
                    model_mask=self.model_mask,
                    optim=optim,
                    theta=theta,
                    grid=grid,
                    array=array,
                    optim_state=optim_state,
                    lr_schedule_state=lr_schedule_state,
                    saveat=saveat,
                    poly_project=poly_project,
                    alpha_reg=alpha_reg,
                )

                # Save the loss of this batch
                loss_step += loss_batch

            # Save the loss of this step
            (self.losses_train).append(loss_step / n_batches)

            # Save the best model so far in the training
            if self.losses_train[-1] < self.losses_best[0]:
                diff_model = eqx.filter(self.model[0], self.model_mask)
                eqx.tree_serialise_leaves(
                    "Outputs/Model_training_%s.eqx" % (suffix), diff_model
                )
                self.losses_best[0] = self.losses_train[-1]

            # Adjusts the learning rate scaling value
            diff_model = eqx.filter(self.model[0], self.model_mask)
            _, lr_schedule_state = lr_schedule.update(
                updates=diff_model, state=lr_schedule_state, value=lr_loss
            )
            self.lr_history.append(lr_schedule_state.scale)

            # Print partial results and save the models
            if step % print_every == 0:
                if self.n_validation > 0:
                    print(
                        "Epoch = %d, Loss_training = %.4f, Loss_validation = %.4f"
                        % (step + 1, self.losses_train[-1], self.losses_validation[-1])
                    )

                    # Save losses
                    f = h5.File("Outputs/Losses_%s.h5" % (suffix), "w")
                    f.create_dataset("loss_training", data=self.losses_train)
                    f.create_dataset("loss_validation", data=self.losses_validation)
                    f.create_dataset("lr_history", data=self.lr_history)
                    f.close()

                else:
                    print(
                        "Epoch = %d, Loss_training = %.4f"
                        % (step + 1, self.losses_train[-1])
                    )

                    # Save losses
                    f = h5.File("Outputs/Losses_%s.h5" % (suffix), "w")
                    f.create_dataset("loss_training", data=self.losses_train)
                    f.create_dataset("lr_history", data=self.lr_history)
                    f.close()

            # Stop the loop if the learning rate got very small
            if lr_schedule_state.scale <= lr_limit:
                print("The CNF converged!")
                break

        # Return the states
        return optim_state, lr_schedule_state

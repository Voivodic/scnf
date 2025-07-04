"""
Test the inference module.
"""

# Import the core modules
import e3nn_jax as e3nn
import optax
import jax.numpy as jnp
import jax.random as jrandom
import pytest
from jaxtyping import Array, Float

# Import the module with the cnf
from scnf import inference, cnf


# Define a function to transform a grid under E(3)
def grid_transform(
    grid: Float[Array, "ND ND ND"],
    rotations: list = [0, 0, 0],
    shifts: list = [0, 0, 0],
) -> jnp.ndarray:
    """Transform a grid under E(3) operations (rotations and shifts).

    This function applies a series of 90-degree rotations and cyclic shifts
    to a 3D JAX array (grid), simulating E(3) transformations.

    :param grid: The input 3D JAX array (grid) to be transformed.
    :type grid: Float[Array, "ND ND ND"]
    :param rotations: A list of three integers [k0, k1, k2] specifying the
        number of 90-degree rotations to apply around axes (1,2), (0,2), and (0,1)
        respectively. Defaults to [0, 0, 0] (no rotations).
    :type rotations: list
    :param shifts: A list of three integers [s0, s1, s2] specifying the
        number of cyclic shifts to apply along each of the three axes.
        Defaults to [0, 0, 0] (no shifts).
    :type shifts: list

    :returns: The transformed 3D JAX array.
    :rtype: jnp.ndarray
    """
    # Do not transform inplace
    transformed_grid = grid.copy()

    # # Apply the shifts
    for axis in range(3):
        transformed_grid = jnp.roll(transformed_grid, shift=shifts[axis], axis=axis)

    # Apply the rotations
    transformed_grid = jnp.rot90(transformed_grid, k=rotations[0], axes=(1, 2))
    transformed_grid = jnp.rot90(transformed_grid, k=rotations[1], axes=(0, 2))
    transformed_grid = jnp.rot90(transformed_grid, k=rotations[2], axes=(0, 1))

    return transformed_grid


# --- Fixtures for test setup ---


@pytest.fixture(scope="module")
def test_params():
    """Defines common parameters for all tests."""
    return {
        "ND": 8,
        "N_GRIDS": 1000,
        "SEED": 12345,
        "CONV_IRREPS": [
            "1x0e",
            "4x0e",
            "11x0e+1x2e",
            "19x0e+1x1o+2x2e",
            "64x0e",
        ],
        "CONV_CHANNELS": [1, 4, 16, 32, 64],
        "KERNEL_SIZE": 3,
        "KERNEL_FOURIER_SIZE": 32,
        "STRIDE": 1,
        "CELL_SIZE": 1.0,
        "N_NEURONS": [3, 8, 8, 3],
        "N_NEURONS_LINS": [64, 32, 16, 8],
        "N_NEURONS_RADIAL": [4, 4],
        "N_NEURONS_ARRAY": [32, 16, 8],
        "POOLING_SIZE": 1,
        "KERNEL_POOLING_SIZE": 1,
        "TIME_SIZE": 100,
        "TOLERANCE": 1e-5,
        "KERNEL_POOLING_TYPE": "exp",
        "N_TIMES": 100,
        "CONV_SPACE": "configuration",
        "T0": 0.0,
        "T1": 1.0,
        "DT0": 0.1,
        "N_TRAIN": 900,
        "N_VALIDATION": 100,
        "LEARNING_RATE": 1e-3,
        "N_EPOCHS": 10,
        "BATCH_SIZE": 50,
        "POLY_ORDER": 2,
        "ALPHA_REG": 1.0,
    }


@pytest.fixture(scope="module")
def rng_key(test_params):
    """Provides a JAX PRNG key."""
    return jrandom.PRNGKey(test_params["SEED"])


@pytest.fixture(scope="module")
def times(test_params):
    return jnp.linspace(0.0, 1.0, test_params["N_TIMES"])


@pytest.fixture(scope="module")
def thetas(rng_key, test_params):
    return jrandom.normal(
        rng_key, shape=(test_params["N_GRIDS"], test_params["N_NEURONS"][0])
    )


@pytest.fixture(scope="module")
def rotation_and_shift_arrays(rng_key, test_params):
    """Generates random rotation and shift arrays."""
    key, key_rotation, key_shift = jrandom.split(rng_key, num=3)
    rotations = jrandom.randint(key_rotation, (test_params["N_GRIDS"], 3), 0, 4)
    shifts = jrandom.randint(
        key_shift,
        (test_params["N_GRIDS"], 3),
        -test_params["ND"] // 2,
        test_params["ND"] // 2 + 1,
    )

    return rotations, shifts


@pytest.fixture(scope="module")
def input_grids(rng_key, rotation_and_shift_arrays, test_params):
    """Creates initial and transformed grids."""
    key, _ = jrandom.split(rng_key)
    rotations, shifts = rotation_and_shift_arrays

    grid1_array = jrandom.normal(
        key,
        shape=(
            test_params["N_GRIDS"],
            test_params["ND"],
            test_params["ND"],
            test_params["ND"],
        ),
    )
    grid1 = e3nn.IrrepsArray(
        irreps=test_params["CONV_IRREPS"][0], array=grid1_array[:, :, :, :, jnp.newaxis]
    )

    grid2_list = []
    for i in range(test_params["N_GRIDS"]):
        grid2_list.append(
            grid_transform(
                grid1.array[i, :, :, :, 0],
                rotations=rotations[i].tolist(),
                shifts=shifts[i].tolist(),
            )
        )
    grid2 = jnp.array(grid2_list)
    grid2 = e3nn.IrrepsArray(
        irreps=test_params["CONV_IRREPS"][0], array=grid2[:, :, :, :, jnp.newaxis]
    )

    return grid1, grid2


@pytest.fixture(scope="module")
def input_arrays(rng_key, test_params):
    """Creates initial and transformed grids."""
    key, _ = jrandom.split(rng_key)
    array = jrandom.normal(
        key,
        shape=(
            test_params["N_GRIDS"],
            test_params["N_NEURONS_ARRAY"][0],
        ),
    )

    return array


@pytest.fixture(scope="module")
def optim(test_params):
    """Initializes the optimizer."""
    return optax.adam(test_params["LEARNING_RATE"])


@pytest.fixture(scope="module")
def cnf_class(rng_key, test_params):
    """Initializes a CNF."""
    vf = cnf.cnf(
        key=rng_key,
        n_neurons=test_params["N_NEURONS"],
        grid_size=[test_params["ND"], test_params["ND"], test_params["ND"]],
        kernel_size=test_params["KERNEL_SIZE"],
        cell_size=test_params["CELL_SIZE"],
        conv_irreps=test_params["CONV_IRREPS"],
        n_neurons_lins=test_params["N_NEURONS_LINS"],
        n_neurons_radial=test_params["N_NEURONS_RADIAL"],
        n_neurons_array=test_params["N_NEURONS_ARRAY"],
        pooling_stride=test_params["POOLING_SIZE"],
        kernel_pooling_size=test_params["KERNEL_POOLING_SIZE"],
        n_channels=test_params["CONV_CHANNELS"],
        conv_space=test_params["CONV_SPACE"],
        t0=test_params["T0"],
        t1=test_params["T1"],
        dt0=test_params["DT0"],
    )
    vf.compute_kernels()

    return vf


@pytest.fixture(scope="module")
def cnf_array_class(rng_key, test_params):
    """Initializes a CNF."""
    vf = cnf.cnf(
        key=rng_key,
        n_neurons=test_params["N_NEURONS"],
        grid_size=[],
        kernel_size=test_params["KERNEL_SIZE"],
        cell_size=test_params["CELL_SIZE"],
        conv_irreps=[],
        n_neurons_lins=[],
        n_neurons_radial=[],
        n_neurons_array=test_params["N_NEURONS_ARRAY"],
        pooling_stride=test_params["POOLING_SIZE"],
        kernel_pooling_size=test_params["KERNEL_POOLING_SIZE"],
        n_channels=[],
        conv_space=test_params["CONV_SPACE"],
        t0=test_params["T0"],
        t1=test_params["T1"],
        dt0=test_params["DT0"],
    )
    vf.compute_kernels()

    return vf


@pytest.fixture(scope="module")
def cnf_unconditional_class(rng_key, test_params):
    """Initializes a CNF."""
    vf = cnf.cnf(
        key=rng_key,
        n_neurons=test_params["N_NEURONS"],
        grid_size=[],
        kernel_size=test_params["KERNEL_SIZE"],
        cell_size=test_params["CELL_SIZE"],
        conv_irreps=[],
        n_neurons_lins=[],
        n_neurons_radial=[],
        n_neurons_array=[],
        pooling_stride=test_params["POOLING_SIZE"],
        kernel_pooling_size=test_params["KERNEL_POOLING_SIZE"],
        n_channels=[],
        conv_space=test_params["CONV_SPACE"],
        t0=test_params["T0"],
        t1=test_params["T1"],
        dt0=test_params["DT0"],
    )
    vf.compute_kernels()

    return vf


@pytest.fixture(scope="module")
def inference_unconditional_class(
    rng_key, cnf_unconditional_class, test_params, thetas
):
    """Initializes a CNF."""
    vf = inference.inference(
        key=rng_key,
        n_train=test_params["N_TRAIN"],
        n_validation=test_params["N_VALIDATION"],
        data_loader_theta=lambda ind: thetas[ind, :],
        data_loader_grid=lambda _: jnp.array([]),
        data_loader_array=lambda _: jnp.array([]),
        model=cnf_unconditional_class,
    )

    return vf


@pytest.fixture(scope="module")
def inference_array_class(rng_key, cnf_array_class, test_params, thetas, input_arrays):
    """Initializes a CNF."""
    # Unpack the input data
    array = input_arrays

    # Create the inference class
    vf = inference.inference(
        key=rng_key,
        n_train=test_params["N_TRAIN"],
        n_validation=test_params["N_VALIDATION"],
        data_loader_theta=lambda ind: thetas[ind, :],
        data_loader_grid=lambda _: jnp.array([]),
        data_loader_array=lambda ind: array[ind, :],
        model=cnf_array_class,
    )

    return vf


@pytest.fixture(scope="module")
def inference_class(rng_key, cnf_class, test_params, thetas, input_grids, input_arrays):
    """Initializes a CNF."""
    # Unpack the input data
    grid1, grid2 = input_grids
    array = input_arrays

    # Create the inference class
    vf = inference.inference(
        key=rng_key,
        n_train=test_params["N_TRAIN"],
        n_validation=test_params["N_VALIDATION"],
        data_loader_theta=lambda ind: thetas[ind, :],
        data_loader_grid=lambda ind: grid1[ind, :].array,
        data_loader_array=lambda ind: array[ind, :],
        model=cnf_class,
    )

    return vf


# --- Test Functions ---


def test_logP_validation_unconditional(
    rng_key,
    inference_unconditional_class,
    test_params,
):
    # Create the inference class
    vf = inference_unconditional_class

    # Compute the logp in the validation set
    logP = vf.logP_validation(batch_size=test_params["N_VALIDATION"], key=rng_key)

    assert logP != 0.0


def test_logP_validation_array(
    rng_key,
    inference_array_class,
    test_params,
):
    # Create the inference class
    vf = inference_array_class

    # Compute the logp in the validation set
    logP = vf.logP_validation(batch_size=test_params["N_VALIDATION"], key=rng_key)

    assert logP != 0.0


def test_logP_validation(
    rng_key,
    inference_class,
    test_params,
):
    # Create the inference class
    vf = inference_class

    # Compute the logp in the validation set
    logP = vf.logP_validation(batch_size=test_params["N_VALIDATION"], key=rng_key)

    assert logP != 0.0


def test_train_unconditional(
    rng_key,
    inference_unconditional_class,
    optim,
    test_params,
):
    # Create the inference class
    vf = inference_unconditional_class

    # Train the model
    optim_state, lr_schedule_state = vf.train(
        key=rng_key,
        n_epochs=test_params["N_EPOCHS"],
        batch_size=test_params["BATCH_SIZE"],
        optim=optim,
        print_every=1,
        suffix="test_train_unconditional",
        poly_order=test_params["POLY_ORDER"],
        alpha_reg=test_params["ALPHA_REG"],
    )

    assert optim_state is not None and lr_schedule_state is not None


def test_train_array(
    rng_key,
    inference_array_class,
    optim,
    test_params,
):
    # Create the inference class
    vf = inference_array_class

    # Train the model
    optim_state, lr_schedule_state = vf.train(
        key=rng_key,
        n_epochs=test_params["N_EPOCHS"],
        batch_size=test_params["BATCH_SIZE"],
        optim=optim,
        print_every=1,
        suffix="test_train_unconditional",
        poly_order=test_params["POLY_ORDER"],
        alpha_reg=test_params["ALPHA_REG"],
    )

    assert optim_state is not None and lr_schedule_state is not None


# def test_train(
#     rng_key,
#     inference_class,
#     optim,
#     test_params,
# ):
#     # Create the inference class
#     vf = inference_class
#
#     # Train the model
#     optim_state, lr_schedule_state = vf.train(
#         key=rng_key,
#         n_epochs=test_params["N_EPOCHS"],
#         batch_size=test_params["BATCH_SIZE"],
#         optim=optim,
#         print_every=1,
#         suffix="test_train",
#     )
#
#     assert False

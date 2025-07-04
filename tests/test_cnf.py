"""
Test the continuous normalizing flow (CNF) module.
"""

# Import the core modules
import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest
from jaxtyping import Array, Float

# Import the module with the cnf
from scnf import cnf


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
        "ND": 32,
        "N_GRIDS": 10,
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
        "N_SAMPLES": 1000,
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
def vector_field_unconditional(rng_key, test_params):
    """Initializes a CNF."""
    vf = cnf.vector_field(
        key=rng_key,
        n_neurons=test_params["N_NEURONS"],
        grid_size=[test_params["ND"], test_params["ND"], test_params["ND"]],
        kernel_size=test_params["KERNEL_SIZE"],
        cell_size=test_params["CELL_SIZE"],
        conv_irreps=[],
        n_neurons_lins=[],
        n_neurons_radial=test_params["N_NEURONS_RADIAL"],
        n_neurons_array=[],
        pooling_stride=test_params["POOLING_SIZE"],
        kernel_pooling_size=test_params["KERNEL_POOLING_SIZE"],
        n_channels=[],
        conv_space=test_params["CONV_SPACE"],
    )
    vf.compute_kernels()

    return vf


@pytest.fixture(scope="module")
def vector_field_array(rng_key, test_params):
    """Initializes a CNF."""
    vf = cnf.vector_field(
        key=rng_key,
        n_neurons=test_params["N_NEURONS"],
        grid_size=[test_params["ND"], test_params["ND"], test_params["ND"]],
        kernel_size=test_params["KERNEL_SIZE"],
        cell_size=test_params["CELL_SIZE"],
        conv_irreps=[],
        n_neurons_lins=[],
        n_neurons_radial=test_params["N_NEURONS_RADIAL"],
        n_neurons_array=test_params["N_NEURONS_ARRAY"],
        pooling_stride=test_params["POOLING_SIZE"],
        kernel_pooling_size=test_params["KERNEL_POOLING_SIZE"],
        n_channels=[],
        conv_space=test_params["CONV_SPACE"],
    )
    vf.compute_kernels()

    return vf


@pytest.fixture(scope="module")
def vector_field_grid_non_eq(rng_key, test_params):
    """Initializes a CNF."""
    vf = cnf.vector_field(
        key=rng_key,
        n_neurons=test_params["N_NEURONS"],
        grid_size=[test_params["ND"], test_params["ND"], test_params["ND"]],
        kernel_size=test_params["KERNEL_SIZE"],
        cell_size=test_params["CELL_SIZE"],
        conv_irreps=[],
        n_neurons_lins=test_params["N_NEURONS_LINS"],
        n_neurons_radial=test_params["N_NEURONS_RADIAL"],
        n_neurons_array=[],
        pooling_stride=test_params["POOLING_SIZE"],
        kernel_pooling_size=test_params["KERNEL_POOLING_SIZE"],
        n_channels=test_params["CONV_CHANNELS"],
        conv_space=test_params["CONV_SPACE"],
    )
    vf.compute_kernels()

    return vf


@pytest.fixture(scope="module")
def vector_field_grid_eq_config(rng_key, test_params):
    """Initializes a CNF."""
    vf = cnf.vector_field(
        key=rng_key,
        n_neurons=test_params["N_NEURONS"],
        grid_size=[test_params["ND"], test_params["ND"], test_params["ND"]],
        kernel_size=test_params["KERNEL_SIZE"],
        cell_size=test_params["CELL_SIZE"],
        conv_irreps=test_params["CONV_IRREPS"],
        n_neurons_lins=test_params["N_NEURONS_LINS"],
        n_neurons_radial=test_params["N_NEURONS_RADIAL"],
        n_neurons_array=[],
        pooling_stride=test_params["POOLING_SIZE"],
        kernel_pooling_size=test_params["KERNEL_POOLING_SIZE"],
        n_channels=[],
        conv_space="configuration",
    )
    vf.compute_kernels()

    return vf


@pytest.fixture(scope="module")
def vector_field_grid_eq_fourier(rng_key, test_params):
    """Initializes a CNF."""
    vf = cnf.vector_field(
        key=rng_key,
        n_neurons=test_params["N_NEURONS"],
        grid_size=[test_params["ND"], test_params["ND"], test_params["ND"]],
        kernel_size=test_params["KERNEL_SIZE"] + 1,
        cell_size=test_params["CELL_SIZE"],
        conv_irreps=test_params["CONV_IRREPS"],
        n_neurons_lins=test_params["N_NEURONS_LINS"],
        n_neurons_radial=test_params["N_NEURONS_RADIAL"],
        n_neurons_array=[],
        pooling_stride=test_params["POOLING_SIZE"],
        kernel_pooling_size=test_params["KERNEL_POOLING_SIZE"],
        n_channels=[],
        conv_space="fourier",
    )
    vf.compute_kernels()

    return vf


@pytest.fixture(scope="module")
def vector_field_array_grid(rng_key, test_params):
    """Initializes a CNF."""
    vf = cnf.vector_field(
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
    )
    vf.compute_kernels()

    return vf


@pytest.fixture(scope="module")
def mean_std_layer(rng_key, test_params):
    """Initializes a CNF."""
    vf = cnf.mean_std_layer(
        key=rng_key,
        n_neurons_out=2 * test_params["N_NEURONS"][-1],
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
    )
    vf.compute_kernels()

    return vf


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


# --- Test Functions ---


def test_vector_field_unconditional(
    times,
    thetas,
    vector_field_unconditional,
    input_grids,
    input_arrays,
    test_params,
):
    """Tests the output shape of the vector field.

    This test verifies that the `vector_field` function produces an output
    with the expected shape, given various input parameters.

    :param rng_key: JAX PRNG key for random number generation.
    :type rng_key: jax.random.PRNGKey
    :param times: An array of time points.
    :type times: jax.numpy.ndarray
    :param vector_field: The initialized CNF vector field model.
    :type vector_field: cnf.vector_field
    :param input_grids: A tuple containing the initial and transformed input grids.
    :type input_grids: tuple[e3nn.IrrepsArray, e3nn.IrrepsArray]
    :param test_params: Dictionary containing common test parameters like ND, N_GRIDS, etc.
    :type test_params: dict
    """
    # Get the grids
    grid1, grid2 = input_grids

    # Compute the vector field
    vf = vector_field_unconditional

    # Compute the compressions
    compressed_grid = jax.vmap(vf.compress_grid)(grid1.array)
    compressed_array = jax.vmap(vf.compress_array)(input_arrays)

    # Wrap the vector field
    def vector_field_wrapper(t, theta, compressed_grid, compressed_array):
        return vf(t, theta, (compressed_grid, compressed_array))

    # Compute the vector field
    output1 = jax.vmap(
        jax.vmap(vector_field_wrapper, in_axes=(0, None, None, None)),
        in_axes=(None, 0, 0, 0),
    )(times, thetas, compressed_grid, compressed_array)

    assert output1.shape == (
        test_params["N_GRIDS"],
        test_params["N_TIMES"],
        test_params["N_NEURONS"][-1],
    )


def test_vector_field_array(
    times, thetas, vector_field_array, input_grids, input_arrays, test_params
):
    """Tests the output shape of the vector field.

    This test verifies that the `vector_field` function produces an output
    with the expected shape, given various input parameters.

    :param rng_key: JAX PRNG key for random number generation.
    :type rng_key: jax.random.PRNGKey
    :param times: An array of time points.
    :type times: jax.numpy.ndarray
    :param vector_field: The initialized CNF vector field model.
    :type vector_field: cnf.vector_field
    :param input_grids: A tuple containing the initial and transformed input grids.
    :type input_grids: tuple[e3nn.IrrepsArray, e3nn.IrrepsArray]
    :param test_params: Dictionary containing common test parameters like ND, N_GRIDS, etc.
    :type test_params: dict
    """
    # Get the grids
    grid1, grid2 = input_grids

    # Compute the vector field
    vf = vector_field_array

    # Compute the compressions
    compressed_grid = jax.vmap(vf.compress_grid)(grid1.array)
    compressed_array = jax.vmap(vf.compress_array)(input_arrays)

    # Wrap the vector field
    def vector_field_wrapper(t, theta, compressed_grid, compressed_array):
        return vf(t, theta, (compressed_grid, compressed_array))

    # Compute the vector field
    output1 = jax.vmap(
        jax.vmap(vector_field_wrapper, in_axes=(0, None, None, None)),
        in_axes=(None, 0, 0, 0),
    )(times, thetas, compressed_grid, compressed_array)

    assert output1.shape == (
        test_params["N_GRIDS"],
        test_params["N_TIMES"],
        test_params["N_NEURONS"][-1],
    )


def test_vector_field_grid_non_eq(
    times,
    thetas,
    vector_field_grid_non_eq,
    input_grids,
    input_arrays,
    test_params,
):
    """Tests the output shape of the vector field.

    This test verifies that the `vector_field` function produces an output
    with the expected shape, given various input parameters.

    :param rng_key: JAX PRNG key for random number generation.
    :type rng_key: jax.random.PRNGKey
    :param times: An array of time points.
    :type times: jax.numpy.ndarray
    :param vector_field: The initialized CNF vector field model.
    :type vector_field: cnf.vector_field
    :param input_grids: A tuple containing the initial and transformed input grids.
    :type input_grids: tuple[e3nn.IrrepsArray, e3nn.IrrepsArray]
    :param test_params: Dictionary containing common test parameters like ND, N_GRIDS, etc.
    :type test_params: dict
    """
    # Get the grids
    grid1, grid2 = input_grids

    # Compute the vector field
    vf = vector_field_grid_non_eq

    # Compute the compressions
    compressed_grid = jax.vmap(vf.compress_grid)(grid1.array)
    compressed_array = jax.vmap(vf.compress_array)(input_arrays)

    # Wrap the vector field
    def vector_field_wrapper(t, theta, compressed_grid, compressed_array):
        return vf(t, theta, (compressed_grid, compressed_array))

    # Compute the vector field
    output1 = jax.vmap(
        jax.vmap(vector_field_wrapper, in_axes=(0, None, None, None)),
        in_axes=(None, 0, 0, 0),
    )(times, thetas, compressed_grid, compressed_array)

    assert output1.shape == (
        test_params["N_GRIDS"],
        test_params["N_TIMES"],
        test_params["N_NEURONS"][-1],
    )


def test_vector_field_grid_eq_config(
    times,
    thetas,
    vector_field_grid_eq_config,
    input_grids,
    input_arrays,
    test_params,
):
    """Tests the output shape of the vector field.

    This test verifies that the `vector_field` function produces an output
    with the expected shape, given various input parameters.

    :param rng_key: JAX PRNG key for random number generation.
    :type rng_key: jax.random.PRNGKey
    :param times: An array of time points.
    :type times: jax.numpy.ndarray
    :param vector_field: The initialized CNF vector field model.
    :type vector_field: cnf.vector_field
    :param input_grids: A tuple containing the initial and transformed input grids.
    :type input_grids: tuple[e3nn.IrrepsArray, e3nn.IrrepsArray]
    :param test_params: Dictionary containing common test parameters like ND, N_GRIDS, etc.
    :type test_params: dict
    """
    # Get the grids
    grid1, grid2 = input_grids

    # Compute the vector field
    vf = vector_field_grid_eq_config

    # Compute the compressions
    compressed_grid = jax.vmap(vf.compress_grid)(grid1.array)
    compressed_array = jax.vmap(vf.compress_array)(input_arrays)

    # Wrap the vector field
    def vector_field_wrapper(t, theta, compressed_grid, compressed_array):
        return vf(t, theta, (compressed_grid, compressed_array))

    # Compute the vector field
    output1 = jax.vmap(
        jax.vmap(vector_field_wrapper, in_axes=(0, None, None, None)),
        in_axes=(None, 0, 0, 0),
    )(times, thetas, compressed_grid, compressed_array)

    assert output1.shape == (
        test_params["N_GRIDS"],
        test_params["N_TIMES"],
        test_params["N_NEURONS"][-1],
    )


def test_vector_field_grid_eq_fourier(
    times,
    thetas,
    vector_field_grid_eq_fourier,
    input_grids,
    input_arrays,
    test_params,
):
    """Tests the output shape of the vector field.

    This test verifies that the `vector_field` function produces an output
    with the expected shape, given various input parameters.

    :param rng_key: JAX PRNG key for random number generation.
    :type rng_key: jax.random.PRNGKey
    :param times: An array of time points.
    :type times: jax.numpy.ndarray
    :param vector_field: The initialized CNF vector field model.
    :type vector_field: cnf.vector_field
    :param input_grids: A tuple containing the initial and transformed input grids.
    :type input_grids: tuple[e3nn.IrrepsArray, e3nn.IrrepsArray]
    :param test_params: Dictionary containing common test parameters like ND, N_GRIDS, etc.
    :type test_params: dict
    """
    # Get the grids
    grid1, grid2 = input_grids

    # Compute the vector field
    vf = vector_field_grid_eq_fourier

    # Compute the compressions
    compressed_grid = jax.vmap(vf.compress_grid)(grid1.array)
    compressed_array = jax.vmap(vf.compress_array)(input_arrays)

    # Wrap the vector field
    def vector_field_wrapper(t, theta, compressed_grid, compressed_array):
        return vf(t, theta, (compressed_grid, compressed_array))

    # Compute the vector field
    output1 = jax.vmap(
        jax.vmap(vector_field_wrapper, in_axes=(0, None, None, None)),
        in_axes=(None, 0, 0, 0),
    )(times, thetas, compressed_grid, compressed_array)

    assert output1.shape == (
        test_params["N_GRIDS"],
        test_params["N_TIMES"],
        test_params["N_NEURONS"][-1],
    )


def test_vector_field_array_grid(
    times,
    thetas,
    vector_field_array_grid,
    input_grids,
    input_arrays,
    test_params,
):
    """Tests the output shape of the vector field.

    This test verifies that the `vector_field` function produces an output
    with the expected shape, given various input parameters.

    :param rng_key: JAX PRNG key for random number generation.
    :type rng_key: jax.random.PRNGKey
    :param times: An array of time points.
    :type times: jax.numpy.ndarray
    :param vector_field: The initialized CNF vector field model.
    :type vector_field: cnf.vector_field
    :param input_grids: A tuple containing the initial and transformed input grids.
    :type input_grids: tuple[e3nn.IrrepsArray, e3nn.IrrepsArray]
    :param test_params: Dictionary containing common test parameters like ND, N_GRIDS, etc.
    :type test_params: dict
    """
    # Get the grids
    grid1, grid2 = input_grids

    # Compute the vector field
    vf = vector_field_array_grid

    # Compute the compressions
    compressed_grid = jax.vmap(vf.compress_grid)(grid1.array)
    compressed_array = jax.vmap(vf.compress_array)(input_arrays)

    # Wrap the vector field
    def vector_field_wrapper(t, theta, compressed_grid, compressed_array):
        return vf(t, theta, (compressed_grid, compressed_array))

    # Compute the vector field
    output1 = jax.vmap(
        jax.vmap(vector_field_wrapper, in_axes=(0, None, None, None)),
        in_axes=(None, 0, 0, 0),
    )(times, thetas, compressed_grid, compressed_array)

    assert output1.shape == (
        test_params["N_GRIDS"],
        test_params["N_TIMES"],
        test_params["N_NEURONS"][-1],
    )


def test_mean_std_layer(
    mean_std_layer,
    input_grids,
    input_arrays,
    test_params,
):
    """Tests the output shape of the vector field.

    This test verifies that the `vector_field` function produces an output
    with the expected shape, given various input parameters.

    :param rng_key: JAX PRNG key for random number generation.
    :type rng_key: jax.random.PRNGKey
    :param times: An array of time points.
    :type times: jax.numpy.ndarray
    :param vector_field: The initialized CNF vector field model.
    :type vector_field: cnf.vector_field
    :param input_grids: A tuple containing the initial and transformed input grids.
    :type input_grids: tuple[e3nn.IrrepsArray, e3nn.IrrepsArray]
    :param test_params: Dictionary containing common test parameters like ND, N_GRIDS, etc.
    :type test_params: dict
    """
    # Get the grids
    grid1, grid2 = input_grids

    # Compute the vector field
    vf = mean_std_layer

    # Compute the compressions
    compressed_grid = jax.vmap(vf.compress_grid)(grid1.array)
    compressed_array = jax.vmap(vf.compress_array)(input_arrays)

    # Compute the vector field
    output1 = jax.vmap(vf, in_axes=(0, 0))(compressed_grid, compressed_array)

    assert output1.shape == (
        test_params["N_GRIDS"],
        2 * test_params["N_NEURONS"][-1],
    )


def test_logP_unconditional(
    rng_key,
    times,
    thetas,
    cnf_unconditional_class,
    test_params,
):
    """Tests the output shape of the vector field.

    This test verifies that the `vector_field` function produces an output
    with the expected shape, given various input parameters.

    :param rng_key: JAX PRNG key for random number generation.
    :type rng_key: jax.random.PRNGKey
    :param times: An array of time points.
    :type times: jax.numpy.ndarray
    :param vector_field: The initialized CNF vector field model.
    :type vector_field: cnf.vector_field
    :param input_grids: A tuple containing the initial and transformed input grids.
    :type input_grids: tuple[e3nn.IrrepsArray, e3nn.IrrepsArray]
    :param test_params: Dictionary containing common test parameters like ND, N_GRIDS, etc.
    :type test_params: dict
    """
    # Compute the vector field
    vf = cnf_unconditional_class

    # Set up time points
    saveat = vf.get_saveat(n_times=test_params["N_TIMES"], reverse=True)

    # Compute the logP
    logP = jax.vmap(vf.get_logP, in_axes=(None, 0, None, None, None))(
        rng_key,
        thetas,
        [],
        [],
        saveat,
    )

    assert logP[0].shape == (
        test_params["N_GRIDS"],
        test_params["N_TIMES"],
        test_params["N_NEURONS"][-1],
    )


def test_logP(
    rng_key,
    times,
    thetas,
    cnf_class,
    input_grids,
    input_arrays,
    test_params,
):
    """Tests the output shape of the vector field.

    This test verifies that the `vector_field` function produces an output
    with the expected shape, given various input parameters.

    :param rng_key: JAX PRNG key for random number generation.
    :type rng_key: jax.random.PRNGKey
    :param times: An array of time points.
    :type times: jax.numpy.ndarray
    :param vector_field: The initialized CNF vector field model.
    :type vector_field: cnf.vector_field
    :param input_grids: A tuple containing the initial and transformed input grids.
    :type input_grids: tuple[e3nn.IrrepsArray, e3nn.IrrepsArray]
    :param test_params: Dictionary containing common test parameters like ND, N_GRIDS, etc.
    :type test_params: dict
    """
    # Get the grids
    grid1, grid2 = input_grids

    # Compute the vector field
    vf = cnf_class

    # Set up time points
    saveat = vf.get_saveat(n_times=test_params["N_TIMES"], reverse=True)

    # Compute the logP
    logP = jax.vmap(vf.get_logP, in_axes=(None, 0, 0, 0, None))(
        rng_key,
        thetas,
        grid1.array,
        input_arrays,
        saveat,
    )

    assert logP[0].shape == (
        test_params["N_GRIDS"],
        test_params["N_TIMES"],
        test_params["N_NEURONS"][-1],
    )


def test_sample_unconditional(
    rng_key,
    times,
    cnf_unconditional_class,
    test_params,
):
    """Tests the output shape of the vector field.

    This test verifies that the `vector_field` function produces an output
    with the expected shape, given various input parameters.

    :param rng_key: JAX PRNG key for random number generation.
    :type rng_key: jax.random.PRNGKey
    :param times: An array of time points.
    :type times: jax.numpy.ndarray
    :param vector_field: The initialized CNF vector field model.
    :type vector_field: cnf.vector_field
    :param input_grids: A tuple containing the initial and transformed input grids.
    :type input_grids: tuple[e3nn.IrrepsArray, e3nn.IrrepsArray]
    :param test_params: Dictionary containing common test parameters like ND, N_GRIDS, etc.
    :type test_params: dict
    """
    # Compute the vector field
    vf = cnf_unconditional_class

    # Sample from the CNF
    samples = vf.sample(
        key=rng_key,
        n_samples=test_params["N_SAMPLES"],
        prior=lambda _: 1,
        n_max=100,
        n_times=test_params["N_TIMES"],
    )

    assert samples.shape == (
        test_params["N_SAMPLES"],
        test_params["N_TIMES"],
        test_params["N_NEURONS"][-1],
    )


def test_sample(
    rng_key,
    times,
    cnf_class,
    input_grids,
    input_arrays,
    test_params,
):
    """Tests the output shape of the vector field.

    This test verifies that the `vector_field` function produces an output
    with the expected shape, given various input parameters.

    :param rng_key: JAX PRNG key for random number generation.
    :type rng_key: jax.random.PRNGKey
    :param times: An array of time points.
    :type times: jax.numpy.ndarray
    :param vector_field: The initialized CNF vector field model.
    :type vector_field: cnf.vector_field
    :param input_grids: A tuple containing the initial and transformed input grids.
    :type input_grids: tuple[e3nn.IrrepsArray, e3nn.IrrepsArray]
    :param test_params: Dictionary containing common test parameters like ND, N_GRIDS, etc.
    :type test_params: dict
    """
    # Get the grids
    grid1, grid2 = input_grids

    # Compute the vector field
    vf = cnf_class

    # Sample from the CNF
    samples = vf.sample(
        key=rng_key,
        n_samples=test_params["N_SAMPLES"],
        grid=grid1[0, :].array,
        array=input_arrays[0, :],
        prior=lambda _: 1,
        n_max=100,
        n_times=test_params["N_TIMES"],
    )

    assert samples.shape == (
        test_params["N_SAMPLES"],
        test_params["N_TIMES"],
        test_params["N_NEURONS"][-1],
    )

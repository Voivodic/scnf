import jax
import jax.numpy as jnp
import h5py as h5

def generate_data(N, D, P, key):
    """Generates N D-dimensional random numbers with non-Gaussian distribution.

    :param N: Number of samples.
    :type N: int
    :param D: Number of dimensions.
    :type D: int
    :param P: Order of the polynomial transformation.
    :type P: int
    :param key: JAX random key.
    :type key: jax.random.PRNGKey
    :return: An array of shape (N, D) with the generated data.
    :rtype: jnp.ndarray
    """
    # Split the key for different random operations
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)

    # 1. Generate Gaussian-distributed random numbers
    gaussian_samples = jax.random.normal(subkey1, shape=(N, D))

    # 2. Generate a random rotation matrix
    # Using QR decomposition of a random matrix to get a random orthogonal matrix
    random_matrix = jax.random.normal(subkey2, shape=(D, D))
    rotation_matrix, _ = jnp.linalg.qr(random_matrix)

    # Apply the rotation
    correlated_samples = jnp.dot(gaussian_samples, rotation_matrix)

    # 3. Apply a polynomial transformation
    # Generate random polynomial coefficients
    poly_coeffs = jax.random.normal(subkey3, shape=(D, P + 1))

    # Apply the polynomial transformation to each dimension
    transformed_samples = jnp.zeros_like(correlated_samples)
    for i in range(D):
        p = jnp.poly1d(poly_coeffs[i])
        transformed_samples = transformed_samples.at[:, i].set(p(correlated_samples[:, i]))

    return transformed_samples

if __name__ == '__main__':
    # Example usage
    N = 1000  # Number of samples
    D = 2     # Number of dimensions
    P = 3     # Polynomial order

    # Create a random key
    key = jax.random.PRNGKey(12345)

    # Generate the data
    data = generate_data(N, D, P, key)

    # Print some information about the generated data
    print(f"Generated data shape: {data.shape}")

    # Save the data to a HDF5  
    with h5.File("Data/thetas.hdf5", "w") as f:
        f.create_dataset("data", data=data)

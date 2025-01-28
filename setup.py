from setuptools import setup

# Run the setup script to build the C extensions and install the package
setup(
    name="scnf",
    version="0.1.0",
    packages=["scnf"],
    install_requires=[
        "jax",
        "equinox",
        "diffrax",
        "jaxtyping",
        "typing",
        "optax",
        "numpy",
        "tqdm",
        "e3nn_jax",
    ],
)

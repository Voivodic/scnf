# SCNF

Python library for Steerable Continuous Normalizing Flows (JAX + Equinox + Diffrax + e3nn_jax).

## Commands

All commands assume Nix. No packages are installed on the host.

- **Run all tests**: `nix run .#test`
- **Run a single test file**: `nix run .#test -- tests/test_layers.py`
- **Lint** (inside dev shell): `nix develop -c ruff check .`
- **Typecheck** (inside dev shell): `nix develop -c pyright scnf/`
- **Build package** (runs tests as check): `nix build .`
- **Python REPL with scnf**: `nix run .`
- **Dev shell**: `nix develop`

Tests are JIT-compiled and ODE-based; they are not fast. `nix run .#test` sets `JAX_PLATFORMS=cpu` automatically.

## Architecture

The installable package is `scnf/` (declared in `pyproject.toml`). Three modules:

- **`scnf/layers.py`** — Compression layers for grid and array data:
  - `compress_3d_e3` / `compress_fourier_3d_e3`: E(3)-equivariant (use `e3nn_jax`, Clebsch-Gordan, spherical harmonics)
  - `compress_nd`: Non-equivariant CNN compression
  - `compress_array`: Linear compression for 1D arrays
  - `concat_layer`: Time-conditioned concatenation layer
  - `pool_e3_layer`: E(3)-aware pooling
- **`scnf/cnf.py`** — `cnf` class: the continuous normalizing flow model
  - `vector_field_layer`, `mean_std_layer`: internal sub-modules
  - `get_logP()`: backward ODE for log-probability (training)
  - `sample()`: forward ODE + rejection sampling
  - `get_mask()`: separates trainable params from pre-computed kernels (kernels, r_grid, k2 are excluded)
- **`scnf/inference.py`** — `inference` class: training loop
  - `loss()`, `make_step()`: loss + JIT-compiled gradient step
  - `train()`: epoch loop with validation, LR scheduling, checkpointing to `.eqx` files

Root-level `NN.py` and `Fit_grid_CNF.py` are **legacy standalone scripts**, not part of the `scnf` package. `applications/` contains example training/sampling scripts.

## Key Constraints

- `JAX_PLATFORMS=cpu` is **required** for tests and non-GPU runs (set in `flake.nix` check phase)
- E(3) layers **must** call `.compute_kernels()` before forward pass — kernels are recomputed from weights, not stored as trainable params
- Kernel parity: **odd** for `conv_space="configuration"`, **even** for `conv_space="fourier"`
- Vector field `n_neurons` must satisfy `n_neurons[0] == n_neurons[-1]` (same input/output dim)
- Compression layers with equivariant convolutions require that the last conv irreps are all scalars (`lmax == 0`) and match `n_neurons_lins[0]`

## Tooling Config

- **ruff lint**: ignores `F722` and `F821` (needed for jaxtyping shape annotations in type hints)
- **pyright**: `typeCheckingMode = "strict"`, with `reportUnknownMemberType` and `reportMissingTypeStubs` off
- Build: `setuptools` via `pyproject.toml`, package is `scnf` only
- Dev dependencies (pytest, ruff) come from the Nix flake, not `pyproject.toml`

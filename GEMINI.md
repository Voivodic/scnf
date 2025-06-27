# Python library: Continuous normalizing flows with E(3)-steerable convolutional neural networks

This document provides instructions and context for Gemini to assist in the development of this Python library. By following these guidelines, you can help me write clean, well-documented, and thoroughly tested code.

## 1. Project Overview

**Library Name:** Steerable continuous normalizing flows.
**Description:** A python library for performing inference using continuous normalizing flows with E(3)-steerable convolutional neural networks.
**Target Audience:** Researches.
**Key Features:**
  - Feature A: Uses continuous normalizing flows for inference.
  - Feature B: Uses E(3)-steerable convolutional neural networks for the compression of 3D grids.
  - Feature C: Works with both arrays (objects without a symmetry group) and grids.

## 2. Project Structure

The project follows a standard Python library layout:

.
├── GEMINI.md
├── NN.py
├── pyproject.toml
├── README.md
├── scnf
│   ├── cnf.py
│   ├── __init__.py
│   └── layers.py
└── test
    ├── flake.lock
    ├── flake.nix
    ├── test_cnf.py
    └── test_layers.py


When generating new code, please adhere to this structure. For example, new features should be in their own modules within the `scnf` directory, and corresponding tests should be in the `test` directory.

## 3. Coding Style and Conventions

- **Language:** Python 3.13+
- **Style Guide:** Follow PEP 8 for all Python code.
- **Docstrings:** Use Sphinx's format for all modules, classes, and functions.
- **Type Hinting:** All function signatures and variable declarations should include type hints.
- **Imports:** Organize imports into three sections: standard library, third-party libraries, and local application imports, sorted alphabetically within each section.
- **Dependencies:** Please use well-established and actively maintained third-party libraries. If you suggest adding a new dependency, please state the reason.

## 4. Persona

Please act as an expert Python developer and a helpful coding assistant. Be proactive in suggesting improvements and best practices.

By following these instructions, you will be an invaluable partner in building this library. Thank you!

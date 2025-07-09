# Molecode Utils – Agents & Contributor Guide

**Molecode Utils** is a Python utility library for working with the MoleCode dataset. It provides convenience classes and functions to load the MoleCode HDF5 archive, inspect molecular and reaction data, and experiment with simple *Marcus*-type predictive models. This document outlines the project’s structure and the guidelines for contributors, including coding conventions (formatting, docstrings, type hints) and best practices for pull requests. It also describes each module’s role (as an “agent” in the library’s design) and how to set up a development environment consistent with these conventions.

## Project Overview

**Purpose:** Molecode Utils contains reusable utilities and data classes accompanying the MoleCode dataset. The library’s goal is to make it easy to access and manipulate chemical reaction data stored in an HDF5 archive, and to evaluate simple kinetic models on that data. Typical usage includes reading the dataset, filtering reactions, examining molecule properties (with units), and computing predicted reaction barriers using included models.

**Repository Structure:** The project is organized as a standard Python package with additional resources:

- **`src/molecode_utils/`** – the main Python package directory containing all source modules.
- **`examples/`** – a collection of tutorial scripts demonstrating how to use the library’s API in practice.
- **`data/`** – a sample MoleCode HDF5 file for experimentation.
- **`INSTALL.md`** – setup instructions for installing the package.
- **`documentation/`** – module level cheat sheets with code examples.

## Module Responsibilities (Agents)

Each module in the `molecode_utils` package has a specific responsibility – you can think of each as an **agent** specialized in a certain task within the library:

- **`dataset.py`** – *Data access agent*. Defines the `Dataset` class to load the HDF5 archive, filter reactions by criteria, and export data to pandas DataFrames.
- **`filter.py`** – *Filtering agent*. Implements the `Filter` helper, which encapsulates common filtering options.
- **`molecule.py`** – *Molecule agent*. Defines an immutable `Molecule` class with unit-aware values.
- **`reaction.py`** – *Reaction agent*. Defines an immutable `Reaction` class that links two `Molecule` instances.
- **`model.py`** – *Modeling agent*. Provides model classes (`ModelS`, `ModelM1`, etc.) that predict reaction barrier heights.
- **`constants.py`** – *Constants agent*. Houses physical constants and unit conversion factors.

## Code Style and Conventions

- **PEP 8:** Follow standard Python naming and structure guidelines.
- **Black:** Use `black` for code formatting. Install via `pip install black` and run with `black .`.
- **Numpydoc:** All public classes/functions must use Numpydoc-style docstrings.
- **Type Hinting:** All code must include type hints using Python's typing system.

## Pull Request Guidelines

- Use clear titles and descriptions.
- Link issues when applicable (e.g., "Closes #123").
- Ensure all CI checks pass.
- Keep PRs focused and atomic.
- Be responsive to review feedback.

## Development Environment Setup

1. Use Python 3.8+ and a virtual environment.
2. Clone the repo and install with `pip install -e .`
3. Install dev tools: `pip install black mypy`
4. Optionally configure your editor to autoformat with Black.
5. Run `black .` and `mypy .` before committing.
6. Validate functionality using the `examples/` scripts.

Additional module usage examples live under `documentation/`. When adding new
features or modifying behaviour, update the relevant markdown file so that the
cheat sheets stay accurate.

---
Following this **Agents & Contributor Guide** helps maintain the quality and consistency of Molecode Utils. Thank you for contributing!
# molecode_utils Package Overview

This directory contains the Python implementation of the **molecode_utils**
package. It provides a collection of helper classes for working with the
MoleCode HDF5 dataset as well as several simple predictive models.

All modules follow PEP&nbsp;8 formatting, use type hints and include
Numpydoc‑style docstrings.

## Modules

- **`dataset.py`** – defines `Dataset` and `MolecodeArchive` for reading the HDF5 archive, filtering reactions and exporting data to pandas.
- **`filter.py`** – small helper encapsulating common filtering options for `Dataset`.
- **`molecule.py`** – immutable `Molecule` container storing molecule rows with unit‑aware values.
- **`reaction.py`** – immutable `Reaction` container referencing two `Molecule` instances and exposing reaction level data.
- **`model.py`** – base `Model` abstraction and Marcus‑type implementations (`ModelS`, `ModelM1`, `ModelM2`, `ModelM3`, `ModelM4`).
- **`constants.py`** – physical constants and unit conversions used across the codebase.

Each module is heavily documented inline; refer to the source for details and usage examples. Tutorial scripts demonstrating typical workflows live under `../examples/`.

For a high level tour of the API see the cheat sheets in
`../documentation/`.


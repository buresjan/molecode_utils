# Installation

This project is distributed as a normal Python package but is not published to PyPI.
Clone the repository and install it with `pip`.

## Editable mode (for development)

Use the `-e` flag so changes to the source are reflected immediately:

```bash
pip install -e .
```

## Production mode

Install a static copy of the package:

```bash
pip install .
```

Both commands must be executed in the repository root (the directory that
contains `pyproject.toml`).

## Using the package in other projects

The utilities are intended to be reused across analysis repositories. Since
the project will never be published on PyPI you have to install it from a
local clone. Inside the other project's virtual environment run:

```bash
pip install -e /path/to/molecode_utils
```

Replace `/path/to/molecode_utils` with the location of this repository. Omit
the `-e` flag if you want a static install. Afterwards the modules can be
imported normally:

```python
from molecode_utils.dataset import Dataset
```

You can also add the repository as a Git submodule and install it from the
submodule path using the same command.


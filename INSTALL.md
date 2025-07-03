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


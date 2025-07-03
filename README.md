# Molecode Utils

Utilities and data classes for the [MoleCode](https://example.com) dataset. The package ships a small convenience API for loading the HDF5 archive, inspecting molecules and reactions, and experimenting with simple Marcus‑type models.

## Project layout

```
├── data/        # example MoleCode HDF5 archive
├── examples/    # tutorial scripts showcasing functionality
├── src/         # Python package implementation
│   └── molecode_utils/
└── INSTALL.md   # installation instructions
```

See [INSTALL.md](INSTALL.md) for installation instructions.

- The `src/molecode_utils/` directory contains the package sources. A short overview of each module is available in [`src/molecode_utils/README.md`](src/molecode_utils/README.md).
- Tutorial scripts demonstrating typical workflows live in [`examples/`](examples/). Each one is described in detail in [`examples/README.md`](examples/README.md).
- The `data/` directory includes a small sample `.h5` archive for experimentation.

## Getting started

After installing the package you can explore the tutorials. Each script can be executed directly from the project root:

```bash
python examples/dataset_walkthrough.py data/molecode-data-v0.1.0.h5
python examples/molecule_walkthrough.py
python examples/reaction_walkthrough.py
python examples/model_walkthrough.py
```

The tutorials are heavily commented and meant to be run step‑by‑step; they double as documentation for the API.


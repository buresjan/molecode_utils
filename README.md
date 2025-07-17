# Molecode Utils

Utilities and data classes for the [MoleCode](https://example.com) dataset. The
package ships a convenience API for loading the HDF5 archive, inspecting
molecules and reactions, and experimenting with simple Marcus‑type models.

The utilities require **Python&nbsp;3.8+**. For development install the package
in editable mode with `pip install -e .` (see [INSTALL.md](INSTALL.md) for full
instructions).

## Project layout

```
├── data/        # example MoleCode HDF5 archive
├── examples/    # tutorial scripts showcasing functionality
├── src/         # Python package implementation
│   └── molecode_utils/
└── INSTALL.md   # installation instructions
```

Detailed documentation is located in the `documentation/` folder. Start with
[`documentation/README.md`](documentation/README.md) for an index of the
available cheat sheets.

- The `src/molecode_utils/` directory contains the package sources. A short overview of each module is available in [`src/molecode_utils/README.md`](src/molecode_utils/README.md).
- Tutorial scripts demonstrating typical workflows live in [`examples/`](examples/). Each one is described in detail in [`examples/README.md`](examples/README.md).
- The `data/` directory includes a small sample `.h5` archive for experimentation.

## Getting started

After installing the package you can explore the tutorials located in the `examples/` folder.
The tutorials are heavily commented and meant to be run step‑by‑step; they double as documentation for the API.


## Running the dashboard

A small Dash application is included for interactive exploration of the example
MoleCode archive. Install the extra dependencies and start the server:

```bash
pip install dash dash-bootstrap-components pandas plotly h5py
python app.py
```

The app loads `data/molecode-data-v0.1.0.h5` by default and will be available at
`http://127.0.0.1:8050/`.

The window is divided into two columns. The left side hosts the filters and
info tabs, while the right side presents a 2×2 grid of independent figure
panels. Each panel can display a reaction or molecule plot—or a histogram—and
shows the controls relevant to the chosen figure type.

The left column uses a **3:2** vertical split and occupies roughly 40% of the
width, leaving 60% for the figure board. Every quadrant resizes automatically so
no scroll bars appear even when the window is narrowed.

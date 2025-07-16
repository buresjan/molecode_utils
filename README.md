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

After installing the package you can explore the tutorials located in the `examples/` folder.
The tutorials are heavily commented and meant to be run step‑by‑step; they double as documentation for the API.

### Dash integration

When building interactive Dash apps on top of ``molecode_utils``, keep in mind
that Dash disallows periods (``.``) and curly braces in component IDs. Use the
``sanitize_id`` helper or the ``safe_input``/``safe_output`` convenience
wrappers to convert dataset variable names into safe IDs:

```python
from molecode_utils import sanitize_id, safe_input, safe_output

safe_id = sanitize_id("slider-oxidant.0O")  # -> 'slider-oxidant-0O'

# alternatively when defining callbacks
@app.callback(safe_output("slider-oxidant.0O", "value"),
              safe_input("slider-oxidant.0O", "value"))
def on_change(val):
    ...
```

All callbacks and component declarations should use the sanitized ID.


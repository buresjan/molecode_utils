# Tutorial scripts

This folder contains short, self-contained Python scripts that demonstrate how to work with the `molecode_utils` package.
Each file focuses on a specific part of the API and can be run directly from the project root.

- **`dataset_walkthrough.py`** – loads the example archive and showcases Dataset filtering, slicing and exporting features.
- **`molecule_walkthrough.py`** – inspects a single molecule entry and illustrates unit-aware arithmetic.
- **`reaction_walkthrough.py`** – explores reaction data and common access patterns.
- **`model_walkthrough.py`** – runs a few of the included Marcus-type models on sample reactions.

Run a script with Python to follow along, e.g.:

```bash
python examples/dataset_walkthrough.py data/molecode-data-v0.1.0.h5
```

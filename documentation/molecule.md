# Molecule Cheatsheet

`Molecule` is an immutable container representing one row from the molecules table. All fields are `Quantity` objects remembering their units.

## Getting a molecule

```python
mol = dataset.molecule(5)                 # via MolecodeArchive
mol = dataset.reactions_df().iloc[0]['oxidant']  # from a reactions DataFrame
```

Or build manually:

```python
from molecode_utils.molecule import Molecule
rec = h5['molecules'][0]
mol = Molecule.from_row(rec, h5['molecules'].attrs['column_units'])
```

## Accessing fields

```python
print(mol.smiles.value)
print(float(mol.E_H))           # Quantity behaves like a number
```

Dynamic attributes cover all columns. `.info()` prints them nicely:

```python
print(mol.info())
```

Additional helpers:

- `mol.unit('E_H')` – unit string
- `mol.to_dict(include_units=False)` – dictionary of values
- `mol.as_series(include_units=False)` – `pandas.Series`
- `mol.help()` – short REPL tips

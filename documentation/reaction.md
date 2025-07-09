# Reaction Cheatsheet

`Reaction` objects represent a single row from the reactions table together with two `Molecule` participants.
Instances are immutable and expose all reaction columns as attributes of type `Quantity`.

## Constructing reactions

Usually reactions are obtained via `Dataset` or `MolecodeArchive`:

```python
rxn = dataset[0]           # by index
rxn = dataset.get_reaction(42)
```

You can also build one manually using `Reaction.from_row` when you already have the row data and `Molecule` objects:

```python
from molecode_utils.reaction import Reaction
rec = h5['reactions'][0]
rxn = Reaction.from_row(rec, h5['reactions'].attrs['column_units'], molecule_lookup)
```

## Inspecting data

```python
print(rxn.id)
print(rxn.oxidant.smiles)
print(rxn.deltaG0.value, rxn.deltaG0.unit)
```

Use `.info()` for a formatted table of all fields:

```python
print(rxn.info())
```

The `.to_dict()` and `.as_series()` helpers convert the reaction to `dict` or `pandas.Series`.

## Convenience

- `rxn.unit('deltaG0')` – unit string for any field
- `rxn.help()` – short REPL guide (prints to stdout)

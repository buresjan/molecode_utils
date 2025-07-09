# Molecule Cheatsheet

`Molecule` objects provide a convenient, immutable view of the
`molecules` table in the MoleCode archive.  Every column is exposed as a
:class:`Quantity` which remembers its physical unit.
This document walks through typical ways of creating a `Molecule` and
shows how to work with the helper methods.

## Getting a molecule

The easiest route is via :class:`Dataset` or :class:`MolecodeArchive`:

```python
from molecode_utils.dataset import Dataset

with Dataset.from_hdf("data/molecode-data-v0.1.0.h5") as ds:
    mol = ds.molecule(5)           # fetch by mol_idx
    rxn_df = ds.reactions_df()
    mol2 = rxn_df.iloc[0]["oxidant"]  # embedded Molecule
```

`Dataset.reactions_df()` automatically expands the oxidant and substrate
references into full `Molecule` instances so you can grab them directly
from a DataFrame row.

### Manual construction

If you have the raw record and unit information you may construct an
instance yourself:

```python
import h5py
from molecode_utils.molecule import Molecule

with h5py.File("data/molecode-data-v0.1.0.h5") as h5:
    rec = h5["molecules"][0]
    units = h5["molecules"].attrs["column_units"]
    mol = Molecule.from_row(rec, units)
```

## Exploring data fields

`Molecule` exposes every column as a dynamic attribute. Each attribute is
an instance of :class:`Quantity`.

```python
mol.smiles.value      # string without a unit
float(mol.E_H)        # Quantity behaves like a number
mol.dataset           # list of dataset tags (UnitList)
```

The `dataset` column in the archive stores a comma separated list of
originating dataset tags. When parsed by :meth:`from_row`, the value
becomes a :class:`UnitList` which behaves like a normal list but also
carries a ``unit`` attribute.

### Common helpers

* ``mol.unit("E_H")`` – look up the unit string for any field
* ``mol.to_dict()`` – obtain a plain Python ``dict`` of values
* ``mol.to_dict(include_units=True)`` – keep ``(value, unit)`` tuples
* ``mol.as_series()`` – convert to ``pandas.Series`` (same options as
  ``to_dict``)
* ``mol.info()`` – pretty tabular dump of all attributes
* ``mol.help()`` – short reminder of the available utilities

```python
print(mol.info())          # view all values and units
series = mol.as_series()   # easy interoperability with pandas
```

### Alias attributes

Two convenient aliases combine substrate and oxidant specific columns.

```python
mol.target_atom            # first non-missing target atom field
mol.target_atom_other_hs   # same logic for the "other hydrogens" field
```

These attributes choose the value from ``subst_target_atom`` or
``oxid_target_atom`` (and the corresponding ``*_other_hs`` columns)
depending on which one is populated in the source row.

## Practical examples

Below is a collection of small code snippets demonstrating typical
workflows.  Each example assumes you already obtained a ``Molecule``
instance named ``mol`` as shown above.

### Convert to JSON serialisable form

```python
payload = mol.to_dict()
json_str = json.dumps(payload)
```

All numeric ``nan`` values are normalised to ``None`` so the output can be
serialised without additional checks.

### Compute a simple descriptor

```python
avg_redox = (float(mol.E_rad_deprot) + float(mol.E_ox_0)) / 2
print(f"Average redox potential: {avg_redox:.2f} {mol.unit('E_ox_0')}")
```

### Filtering reactions involving the molecule

```python
rdf = dataset.reactions_df()
linked = rdf[(rdf["oxid_idx"] == mol.id) | (rdf["subst_idx"] == mol.id)]
print(f"Molecule {mol.id} participates in {len(linked)} reactions")
```

### Creating a pandas DataFrame from many molecules

```python
mols = [dataset.molecule(i) for i in range(10)]
frame = pd.DataFrame([m.to_dict() for m in mols])
```

### Helpful REPL workflow

When exploring interactively, call ``mol.help()`` to print the short cheat
sheet. The dynamically generated attribute list makes tab completion
in IDEs very handy.

## Summary

`Molecule` is designed to be lightweight yet expressive.  Dynamic
attributes keep the interface compact while helper methods make it easy to
convert to plain Python or pandas objects.  See the tutorial script in
``examples/molecule_tutorial.py`` for a more extensive walkthrough.

# Reaction Cheatsheet

`Reaction` objects represent a single row from the reactions table together with the corresponding oxidant and substrate `Molecule`s. Each reaction property is stored as a `Quantity` which keeps track of the value **and** the physical unit. The class is immutable and easily serialised to dictionaries or pandas objects.

## Constructing reactions

Reactions are usually retrieved from a `Dataset` or `MolecodeArchive` instance:

```python
from molecode_utils.dataset import Dataset

ds = Dataset.from_hdf('data/molecode-data-v0.1.0.h5')
rxn = ds[0]                      # by index
another = ds.get_reaction(42)    # using the reaction id
```

To build a `Reaction` manually use the :pymeth:`Reaction.from_row` helper when you have the raw table row and the required `Molecule` objects:

```python
from molecode_utils.reaction import Reaction

row = h5['reactions'][0]
units = h5['reactions'].attrs['column_units']
rxn = Reaction.from_row(row, units, molecule_lookup)
```

`molecule_lookup` maps ``mol_idx`` values to `Molecule` instances. Missing molecules fall back to minimal stand‑ins containing just the SMILES string.

## Basic inspection

Every column can be accessed as an attribute. The attached unit is accessible via the ``unit`` attribute or through the :py:meth:`Reaction.unit` helper.

```python
print(rxn.id)
print(rxn.oxidant.smiles.value)
print(rxn.deltaG0.value, rxn.deltaG0.unit)
print(rxn.unit('computed_barrier'))
```

Indexing with a column name is equivalent to attribute access:

```python
rxn['deltaG0'] is rxn.deltaG0
```

For a quick overview of all fields call :py:meth:`Reaction.info` which returns a nicely formatted table:

```python
print(rxn.info())
```

## Converting to common formats

`Reaction.to_dict()` and `Reaction.as_series()` turn the object into a plain ``dict`` or ``pandas.Series``. Set ``include_units=True`` to include the units in the output:

```python
as_dict = rxn.to_dict()
with_units = rxn.to_dict(include_units=True)
series = rxn.as_series()
```

These helpers are convenient when exporting data or constructing DataFrames manually.

## Accessing the participants

Both `rxn.oxidant` and `rxn.substrate` are full `Molecule` objects. All molecule descriptors are available as attributes:

```python
print(rxn.oxidant.E_H.value)
print(rxn.substrate.pKaRH.value)
```

You can compute combined quantities easily:

```python
el_diff = rxn.oxidant.E_H.value - rxn.substrate.E_H.value
print(f'E_H difference: {el_diff:.2f} {rxn.oxidant.E_H.unit}')
```

## Working with multiple reactions

`Reaction` instances can be looped over and passed to models or custom functions. Below we compute the average barrier of all reactions that share the same substrate as ``rxn``:

```python
same_subst = [r for r in ds if r.substrate.id == rxn.substrate.id]
barriers = [float(r.computed_barrier) for r in same_subst]
mean_barrier = sum(barriers) / len(barriers)
print(f'Mean barrier for that substrate: {mean_barrier:.2f} kcal/mol')
```

## Immutability

Attempting to modify any attribute raises ``AttributeError``. This guarantees that reactions remain consistent once created:

```python
try:
    rxn.deltaG0 = 0
except AttributeError:
    print('Reactions are immutable')
```

## Building from pandas rows

`Reaction.from_row` also accepts pandas records. This makes it straightforward to recreate objects from a DataFrame:

```python
row = ds.reactions_df().iloc[0]
rxn2 = Reaction.from_row(row, ds.column_units, molecule_lookup)
```

## Quick REPL helpers

The :py:meth:`Reaction.help` message summarises the available convenience methods. It is useful when exploring the object interactively:

```python
rxn.help()
```

``info`` and ``help`` do not require any arguments and print directly to ``stdout``.

## Example: simple barrier difference

The snippet below demonstrates a small calculation using two reactions:

```python
r1 = ds[0]
r2 = ds[1]
barrier_diff = float(r1.computed_barrier) - float(r2.computed_barrier)
print(f'Barrier difference: {barrier_diff:.2f} kcal/mol')
```

## Summary

`Reaction` is a lightweight container that exposes all reaction descriptors with units, links the oxidant and substrate molecules, and provides helpers for converting the data into tabular form. Use it together with `Dataset` to traverse the archive or construct instances manually for custom data processing.

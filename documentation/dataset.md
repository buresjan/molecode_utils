# Dataset & MolecodeArchive Cheatsheet

This document lists everything you can do with `Dataset` and `MolecodeArchive` from `molecode_utils.dataset`.

## Opening the archive

```python
from molecode_utils.dataset import Dataset

# open an archive and automatically close it when done
with Dataset.from_hdf('data/molecode-data-v0.1.0.h5') as ds:
    print(len(ds))       # number of reactions
    print(ds.molecules_df().head())
```

`Dataset.from_hdf()` is a convenience constructor that opens the underlying `MolecodeArchive`. You can also use the lower level context manager directly:

```python
from molecode_utils.dataset import MolecodeArchive

with MolecodeArchive('data/molecode-data-v0.1.0.h5') as arc:
    rxn_df = arc.reactions_df()
    mol_df = arc.molecules_df()
    r9 = arc.reaction(9)
```

## Navigating a Dataset

- `len(ds)` – number of reactions
- `for rxn in ds:` – iterate over `Reaction` objects
- `ds[i]` – random access by index or slice
- `ds.get_reaction(idx)` – retrieve a specific `Reaction` and ensure it belongs to the view
- `ds.add_reaction(rxn)` – add an existing `Reaction` object to the view (no disk I/O)

## Export helpers

- `ds.reactions_df()` – joined DataFrame with reaction and molecule columns
- `ds.molecules_df()` – all molecules referenced by the view
- `ds.describe()` – numeric summary (wraps `pandas.DataFrame.describe`)

```python
rdf = ds.reactions_df(add_dataset_main=True)
print(rdf[['rxn_idx', 'dataset_main', 'computed_barrier']].head())
```

## Filtering

`Dataset.filter` returns a new immutable view. Criteria can be combined:

```python
# pandas style query on reaction columns
subset = ds.filter(query="computed_barrier > 10")

# dataset tag filtering
phenols = ds.filter(datasets=["Phenols"])

# column inequalities
fast = ds.filter(deltaG0__lt=-5, oxidant__E_H__ge=1.0)

# custom lambda over Reaction objects
only_complete = ds.filter(func=lambda r: r.computed_barrier is not None)
```

The `Filter` helper in `molecode_utils.filter` packages common criteria (see `documentation/filter.md`).

## Dynamic column access

Any reaction column can be accessed as an attribute:

```python
barriers = ds.computed_barrier      # pandas Series
```

This works because `Dataset.__getattr__` proxies columns from `reactions_df()`.

## Closing

When `Dataset.from_hdf` is used with `with`, the archive is closed automatically. If you constructed the object manually call `ds.close()` when done.

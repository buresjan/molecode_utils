# Dataset & MolecodeArchive Cheatsheet

This document lists the most useful helpers provided by
`molecode_utils.dataset`.  The focus is on the high level
`Dataset` view which lets you inspect, filter and export reactions stored
inside the MoleCode archive.

## Opening the archive

The easiest way to open the archive is through
`Dataset.from_hdf()` which returns a ready to use view.

```python
from molecode_utils.dataset import Dataset

# open an archive and automatically close it when done
with Dataset.from_hdf("data/molecode-data-v0.1.0.h5") as ds:
    print("total reactions:", len(ds))
    print(ds.molecules_df().head())
```

`Dataset.from_hdf()` internally creates a `MolecodeArchive`.  You can use the
archive directly when you only need raw tables or single reactions:

```python
from molecode_utils.dataset import MolecodeArchive

with MolecodeArchive("data/molecode-data-v0.1.0.h5") as arc:
    rxn_df = arc.reactions_df()        # DataFrame with reaction columns
    mol_df = arc.molecules_df()        # molecules table
    r9 = arc.reaction(9)               # a single Reaction instance
```

## Navigating a Dataset

`Dataset` acts a bit like a Python sequence:

- `len(ds)` – number of reactions
- `for rxn in ds:` – iterate over :class:`Reaction` objects
- `ds[i]` – random access by index (or slice for a new view)
- `ds.get_reaction(idx)` – retrieve a specific reaction and ensure it
  belongs to the current view
- `ds.add_reaction(rxn)` – add an existing :class:`Reaction` object
  to the view (no disk I/O)

```python
rxn0 = ds[0]
rxn_slice = ds[:10]                   # first ten reactions as a new Dataset
specific = ds.get_reaction(rxn0.id)
```

Because views are immutable you can chain slicing and filtering without
modifying the original dataset.

## Export helpers

`Dataset.reactions_df()` joins the reactions table with molecule columns.
`Dataset.molecules_df()` returns only the molecules referenced by the view.
`Dataset.describe()` exposes :func:`pandas.DataFrame.describe` for numeric
columns.

```python
rdf = ds.reactions_df(add_dataset_main=True)
print(rdf[["rxn_idx", "dataset_main", "computed_barrier"]].head())

# save everything to CSV
rdf.to_csv("reactions.csv", index=False)
```

The ``add_dataset_main`` flag is handy for grouping by the first dataset tag:

```python
summary = (
    ds.reactions_df(add_dataset_main=True)
    .groupby("dataset_main")
    ["computed_barrier"]
    .mean()
)
print(summary.sort_values())
```

## Filtering in depth

``Dataset.filter`` returns a **new** view with only the reactions that match
all given criteria.  Several styles can be combined:

```python
# pandas-style query on reaction columns
sub = ds.filter(query="computed_barrier > 10")

# dataset tag filtering (case-insensitive substring match)
phenols = ds.filter(datasets=["Phenols"])

# column inequalities – suffixes: __lt, __le, __gt, __ge, __eq, __ne
fast = ds.filter(deltaG0__lt=-5, oxidant__E_H__ge=1.0)

# custom lambda receiving each Reaction
only_complete = ds.filter(func=lambda r: r.computed_barrier is not None)

# everything together – chainable!
combo = (
    ds.filter(datasets=["Phenols"])  # first narrow down by tag
      .filter(query="deltaG0 < 0")   # then by expression
      .filter(**{"oxidant.E_H__gt": 1.0})
)
```

For reusable logic wrap the arguments in the :class:`Filter` helper
(see ``documentation/filter.md``).

## Working with Reaction objects

Iterating over a dataset yields fully fledged ``Reaction`` instances.  You can
use them for custom analysis or to attach them to another dataset view.

```python
for rxn in ds[:3]:
    print(
        f"{rxn.id:5d}  ΔG0={rxn.deltaG0.value:6.2f} {rxn.deltaG0.unit}"
        f"  ΔG‡={rxn.computed_barrier.value:6.2f} {rxn.computed_barrier.unit}"
    )

# create a tiny view containing only those reactions
tiny = Dataset(ds._arc, [])
for rxn in ds[:3]:
    tiny.add_reaction(rxn)
```

## Descriptive statistics

Use ``Dataset.describe()`` to obtain summary statistics over numeric columns:

```python
descr = ds.describe()
print(descr.loc[:, [c for c in descr.columns if c.startswith("deltaG")]])
```

Combined with filtering this is useful for quick exploratory analysis.

## Dynamic column access

Any reaction column from ``reactions_df`` can be accessed as an attribute on the
dataset.  The returned object is the pandas ``Series`` for that column.

```python
barriers = ds.computed_barrier
unit = ds[0].computed_barrier.unit
print(barriers.describe())
```

This feature allows concise one-liners when plotting:

```python
import matplotlib.pyplot as plt
plt.scatter(ds.deltaG0, ds.computed_barrier, s=8)
plt.xlabel(f"ΔG0 [{ds[0].deltaG0.unit}]")
plt.ylabel(f"ΔG‡ [{unit}]")
```

## Closing

When ``Dataset.from_hdf`` is used with ``with``, the archive is closed
automatically.  When creating the ``Dataset`` and ``MolecodeArchive`` manually
you should call ``close()`` when finished to release the underlying HDF5 file.

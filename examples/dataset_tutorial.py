#!/usr/bin/env python
"""
examples/dataset_tutorial.py
============================

Overview of the :class:`~molecode_utils.dataset.Dataset` helper.  The
script opens the archive, demonstrates a few filtering options and shows
how to export data to pandas.
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import pandas as pd

from molecode_utils.dataset import Dataset

# ---------------------------------------------------------------------
# Configuration – path to the MoleCode archive
# ---------------------------------------------------------------------
H5_PATH = pathlib.Path("data/molecode-data-v0.1.0.h5")

print("=" * 88)
print(f"Loading archive: {H5_PATH.resolve()}\n")

with Dataset.from_hdf(H5_PATH) as ds:
    print(f"Total reactions in archive  : {len(ds)}")
    print(f"Distinct molecules          : {len(ds.molecules_df())}\n")

    # ── Filtering example ──────────────────────────────────────────
    phenols = ds.filter(datasets=["Phenols"])
    print(f"Phenols subset size         : {len(phenols)}\n")

    # ── Simple summary ─────────────────────────────────────────────
    print("Summary statistics (ΔG‡ / ΔG0)")
    descr = phenols.describe()
    print(descr.loc[:, [c for c in descr.columns if c.startswith("deltaG") or c.startswith("computed_barrier")]], "\n")

    # ── First three reactions ─────────────────────────────────────-
    print("First three reactions in subset")
    for r in phenols[:3]:
        print(
            f"Reaction {r.id:5d}  ΔG0={r.deltaG0.value:6.2f} {r.deltaG0.unit}"
            f"  ΔG‡={r.computed_barrier.value:6.2f} {r.computed_barrier.unit}"
        )
    print()

    # ── Pandas interoperability ────────────────────────────────────
    rdf = phenols.reactions_df()
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(rdf["computed_barrier"], bins=30)
    ax.set_xlabel("ΔG‡  [kcal/mol]")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig("examples/assets/phenols_hist.png")
    print("Histogram saved as 'phenols_hist.png'\n")

print("Tutorial complete ✓")
print("=" * 88)

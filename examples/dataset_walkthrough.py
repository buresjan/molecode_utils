#!/usr/bin/env python
"""
examples/dataset_walkthrough.py
===============================

Granular tour of *Dataset* power-features
----------------------------------------
1. Load an entire archive as a Dataset view.
2. Showcase each of the four filter mechanisms separately, then together.
3. Demonstrate chainability, slicing, len(), iteration, add/get helpers.
4. Export to pandas, run quick stats, groupby, plotting (optional).
5. Close everything cleanly.

Run:
    $ python examples/dataset_walkthrough.py [path-to-h5]

The script is meant to be read as much as executed – feel free to tweak &
experiment in a REPL or Jupyter after each section.
"""
from __future__ import annotations

import sys
import pathlib
from textwrap import indent

import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------------------------------------------------
# Make the *src* directory importable when run from repo root
# ----------------------------------------------------------------------
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from molecode_utils.dataset import Dataset

# Optional CLI arg – path to HDF5 file
H5_PATH = pathlib.Path("data/molecode-data-v0.1.0.h5")

print("=" * 88)
print(f"Loading MoleCode archive: {H5_PATH.resolve()}\n")

# ----------------------------------------------------------------------
# 1. Open a *Dataset* view over **all** reactions in the file
# ----------------------------------------------------------------------
with Dataset.from_hdf(H5_PATH) as ds:

    print("Dataset view created            :", ds)    # shows len()
    print("Number of distinct molecules    :", len(ds.molecules_df()), "\n")

    # ------------------------------------------------------------------
    # 2. Filtering – each channel individually
    # ------------------------------------------------------------------
    print("--- Individual filter channels ----------------------------------")

    # 2.1 pandas-style query string
    q1 = ds.filter(query="computed_barrier < 5 and deltaG0 < 0")
    print("query='…'                       :", len(q1))

    # 2.2 dataset tag selector (case-insensitive substrings, any match)
    q2 = ds.filter(datasets=["Phenols_para"])
    print("datasets=['Phenols_para']       :", len(q2))

    # 2.3 column-DSL inequalities (suffix __lt, __gt, …)
    q3 = ds.filter(KED_H__gt=0.6, asynchronicity__lt=30)
    print("column-DSL                      :", len(q3))

    # # 2.4 custom lambda on *Reaction*
    # q4 = ds.filter(func=lambda r: float(r.tunneling_corr_reaction) < -0.5)
    # print("func=lambda r: …               :", len(q4))

    # ------------------------------------------------------------------
    # 3. Chain all filters together  (intersection logic)
    # ------------------------------------------------------------------
    print("\n--- Chaining filters (AND) -------------------------------------")
    small = (
        ds
        .filter(query="computed_barrier < 8 and deltaG0 < 0")
        .filter(datasets=["Phenols_para"])
        .filter(KED_H__ge=0.5)
        .filter(func=lambda r: r.deltaG0_inner.value < -20)
    )
    print("Resulting subset size           :", len(small), "\n")

    # ------------------------------------------------------------------
    # 4. Inspect the subset
    # ------------------------------------------------------------------
    print("--- describe() ---------------------------------------------------")
    print(small.describe(), "\n")

    # quick peek at the first three reactions
    print("--- First three reactions ---------------------------------------")
    for r in small[:3]:
        print(
            f"Reaction {r.id:5d}  ΔG0={r.deltaG0.value:6.2f} {r.deltaG0.unit}"
            f"  ΔG‡={r.computed_barrier.value:6.2f} {r.computed_barrier.unit}"
        )
    print()

    # ------------------------------------------------------------------
    # 5. DataFrame export + vanilla pandas analysis
    # ------------------------------------------------------------------
    print("--- Pandas interoperability -------------------------------------")
    rdf = small.reactions_df()
    print("DataFrame shape                :", rdf.shape)

    # median barrier by individual dataset tag
    tag_med = (
        rdf
        .assign(tag=rdf["datasets_str"].str.split(","))   # ① build list column
        .explode("tag")                                   # ② explode by name
        .groupby("tag", dropna=True)["computed_barrier"]
        .median()
        .sort_values()
        .head(5)
    )

    print("\nTop-5 lowest median ΔG‡ per tag:")
    print(tag_med.to_string())


    # optional mini-plot – comment out if running headless
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(rdf["computed_barrier"], bins=30)
    ax.set_xlabel("ΔG‡  [kcal/mol]")
    ax.set_ylabel("count")
    ax.set_title("Subset barrier distribution")
    fig.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # 6. Working with molecules
    # ------------------------------------------------------------------
    print("--- Molecule table for the subset --------------------------------")
    mdf = small.molecules_df()
    print("Distinct molecules in subset   :", mdf.shape[0])
    print("First 3 SMILES                 :")
    for s in mdf["smiles"].head(3):
        print("  •", s.rstrip())             # HDF5 fixed-width strings have padding
    print()

    # ------------------------------------------------------------------
    # 7. CRUD helpers  (add / get)
    # ------------------------------------------------------------------
    print("--- add_reaction() / get_reaction() -----------------------------")
    some_r = ds[0]              # first reaction overall
    new_view = small            # start from previous subset
    new_view.add_reaction(some_r)
    print("After manual add               :", len(new_view))

    fetched = new_view.get_reaction(some_r.id)
    print("Fetched reaction ΔG‡           :", fetched.computed_barrier.value,
          fetched.computed_barrier.unit, "\n")

    # ------------------------------------------------------------------
    # 8. Combining Dataset views manually (union)
    # ------------------------------------------------------------------
    print("--- Manual union of two views -----------------------------------")
    union_view = Dataset(ds._arc, small._rxn_ids + q2._rxn_ids)   # quick hack
    print("Union size (small ∪ q2)        :", len(union_view), "\n")

    # ------------------------------------------------------------------
    # 9. Slicing / len / iteration speed hint
    # ------------------------------------------------------------------
    print("--- Slicing & iteration -----------------------------------------")
    print("First 5 ids via slice          :", [r.id for r in ds[:5]])
    print("Last id via negative index     :", ds[-1].id)

print("\n✔ Walkthrough finished – file closed automatically")
print("=" * 88)

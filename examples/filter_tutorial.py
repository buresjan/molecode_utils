#!/usr/bin/env python
"""
examples/filter_tutorial.py
===========================

Granular walkthrough of the :class:`molecode_utils.filter.Filter` helper.
The script demonstrates how filters can be composed and reused to
produce subsets of reactions.
"""
from __future__ import annotations

import pathlib

from molecode_utils.dataset import Dataset
from molecode_utils.filter import Filter

H5_PATH = pathlib.Path("data/molecode-data-v0.1.0.h5")

print("=" * 80)
print(f"Loading archive: {H5_PATH.resolve()}\n")

with Dataset.from_hdf(H5_PATH) as ds:
    print(f"Total reactions in archive: {len(ds)}\n")

    # blank filter keeps everything
    keep_all = Filter()
    same = keep_all(ds)
    print(f"Blank filter -> {len(same)} reactions (should match original)\n")

    # reaction level filtering using operator suffixes
    f1 = Filter(reaction={"computed_barrier__gt": 10})
    subset1 = f1(ds)
    print(f"Barrier > 10 kcal/mol -> {len(subset1)} reactions")

    # molecule level filtering using symbolic operators
    f2 = Filter(oxidant={"E_H >=": 1.0})
    subset2 = f2(ds)
    print(f"Oxidant E_H >= 1.0 -> {len(subset2)} reactions")

    # filter reactions originating from a particular dataset
    f3 = Filter(datasets=["Phenols"])
    subset3 = f3(ds)
    print(f"Dataset tag 'Phenols' -> {len(subset3)} reactions")

    # filter on a specific oxidant by molecule identifier
    first_oxid_id = ds.reactions_df().loc[0, "oxidant.molecule_id"]
    f4 = Filter(oxidant={"molecule_id =": first_oxid_id})
    subset4 = f4(ds)
    print(
        f"Oxidant molecule_id == {first_oxid_id.decode()} -> {len(subset4)} reactions"
    )

print("Tutorial complete ✓")
print("=" * 80)

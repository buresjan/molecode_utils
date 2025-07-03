#!/usr/bin/env python
"""
examples/molecule_tutorial.py
=============================

Mini tour of the :class:`~molecode_utils.molecule.Molecule` helper.
The script loads one molecule from the archive and walks through a few
handy features.
"""
from __future__ import annotations

import json
import pathlib
import statistics

import h5py
import pandas as pd

from molecode_utils.molecule import Molecule

# ---------------------------------------------------------------------
# Configuration – adjust the path / mol_idx if needed
# ---------------------------------------------------------------------
H5_PATH = pathlib.Path("data/molecode-data-v0.1.0.h5")
MOL_IDX = 99

print("=" * 72)
print(f"Opening archive: {H5_PATH.resolve()}\n")

with h5py.File(H5_PATH, "r") as h5:
    mol_ds = h5["molecules"]
    units = mol_ds.attrs["column_units"]

    mask = mol_ds["mol_idx"][:] == MOL_IDX
    idx = mask.nonzero()[0]
    if not idx.size:
        raise SystemExit(f"mol_idx {MOL_IDX} not found")
    record = mol_ds[idx[0]]
    mol = Molecule.from_row(record, units)

    rxn_df = pd.DataFrame(h5["reactions"][:])
    units_rxn = json.loads(h5["reactions"].attrs["column_units"])

# ────────────────────────────────────────────────────────────────────
# 1. Quick peek at core attributes
# ────────────────────────────────────────────────────────────────────
print(f"▶ Molecule ID : {mol.id}")
print(f"▶ SMILES      : {mol.smiles.value}")
print(f"▶ Dataset tag : {mol.dataset.value}\n")

print("All descriptors")
print("-" * 34)
for attr in sorted(
    a for a in dir(mol)
    if not a.startswith("_") and a not in {"id"} and not callable(getattr(mol, a))
):
    q = getattr(mol, attr)
    print(f"{attr:30s}: {q.value!s:<20} [{q.unit}]")
print("-" * 34, "\n")

# ────────────────────────────────────────────────────────────────────
# 2. Mini calculation
# ────────────────────────────────────────────────────────────────────
print("Mini-demo calculation")
avg_redox = (float(mol.E_rad_deprot) + float(mol.E_ox_0)) / 2
print(f"  (E_rad_deprot + E_ox_0)/2 = {avg_redox:.3f} V\n")

# ────────────────────────────────────────────────────────────────────
# 3. Reactions involving this molecule
# ────────────────────────────────────────────────────────────────────
linked = rxn_df[(rxn_df["oxid_idx"] == mol.id) | (rxn_df["subst_idx"] == mol.id)]
print(f"▶ Reactions involving molecule-{mol.id}: {len(linked)} found\n")
for _, row in linked.iterrows():
    dg0 = row["deltaG0"]
    dgb = row["computed_barrier"]
    rxn = row["rxn_idx"]
    print(
        f"  Reaction {rxn:4d}: ΔG0 = {dg0:.2f} {units_rxn['deltaG0']},"
        f" ΔG‡ = {dgb:.2f} {units_rxn['computed_barrier']}"
    )

if len(linked):
    med = statistics.median(linked["computed_barrier"])
    print(
        f"\n  Median computed barrier across these reactions: {med:.2f}"
        f" {units_rxn['computed_barrier']}"
    )

print("\nTutorial complete ✓")
print("=" * 72)

#!/usr/bin/env python
"""
examples/reaction_tutorial.py
=============================

Short introduction to the :class:`~molecode_utils.reaction.Reaction`
helper.  One reaction is loaded from the HDF5 archive and a couple of
features are demonstrated.
"""
from __future__ import annotations

import json
import pathlib
import statistics

import h5py
import pandas as pd

from molecode_utils.molecule import Molecule
from molecode_utils.reaction import Reaction

# ---------------------------------------------------------------------
# Configuration – adjust the path / rxn_idx if needed
# ---------------------------------------------------------------------
H5_PATH = pathlib.Path("data/molecode-data-v0.1.0.h5")
RXN_IDX = 9

print("=" * 80)
print(f"Opening archive: {H5_PATH.resolve()}\n")

with h5py.File(H5_PATH, "r") as h5:
    mol_ds = h5["molecules"]
    units_mol = mol_ds.attrs["column_units"]
    mol_df = pd.DataFrame(mol_ds[:])

    mol_lookup = {
        int(row["mol_idx"]): Molecule.from_row(row, units_mol)
        for _, row in mol_df.iterrows()
    }

    rxn_ds = h5["reactions"]
    units_rxn = json.loads(rxn_ds.attrs["column_units"])
    mask = rxn_ds["rxn_idx"][:] == RXN_IDX
    idx = mask.nonzero()[0]
    if not idx.size:
        raise SystemExit(f"rxn_idx {RXN_IDX} not found")
    rec = rxn_ds[idx[0]]
    rxn = Reaction.from_row(rec, units_rxn, molecule_lookup=mol_lookup)

    rxn_df = pd.DataFrame(rxn_ds[:])

# ────────────────────────────────────────────────────────────────────
# 1. Basic info
# ────────────────────────────────────────────────────────────────────
print(f"▶ Reaction ID : {rxn.id}")
print(f"▶ Oxidant     : {rxn.oxidant.smiles.value}")
print(f"▶ Substrate   : {rxn.substrate.smiles.value}\n")

print("All descriptors")
print("-" * 60)
for attr in sorted(
    a for a in dir(rxn)
    if not a.startswith("_")
    and a not in {"id", "oxidant", "substrate"}
    and not callable(getattr(rxn, a))
):
    q = getattr(rxn, attr)
    print(f"{attr:28s}: {q.value!s:<18} [{q.unit}]")
print("-" * 60, "\n")

# ────────────────────────────────────────────────────────────────────
# 2. Mini calculations
# ────────────────────────────────────────────────────────────────────
print("Mini-demo calculations")
dg0 = rxn.deltaG0
dgb = rxn.computed_barrier
delt = float(dgb) - float(dg0)
print(f"  ΔG0           = {dg0.value:.2f} {dg0.unit}")
print(f"  ΔG‡ (barrier) = {dgb.value:.2f} {dgb.unit}")
print(f"  ΔG‡ – ΔG0     = {delt:.2f} {dgb.unit}\n")

# ────────────────────────────────────────────────────────────────────
# 3. Other reactions sharing either participant
# ────────────────────────────────────────────────────────────────────
linked = rxn_df[
    (rxn_df["oxid_idx"] == rxn.oxidant.id)
    | (rxn_df["subst_idx"] == rxn.substrate.id)
]
print(f"▶ Other reactions sharing either participant: {len(linked)} found")
if len(linked):
    med = statistics.median(linked["computed_barrier"])
    print(
        f"  Median computed barrier in that sub-set: {med:.2f}"
        f" {units_rxn['computed_barrier']}"
    )

print("\nTutorial complete ✓")
print("=" * 80)

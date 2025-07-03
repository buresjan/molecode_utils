#!/usr/bin/env python
"""
examples/show_molecule_features.py
==================================

Hands-on tour of *all* `Molecule` capabilities
----------------------------------------------
• Loads one molecule (`mol_idx` given on the CLI or the first in the file).
• Demonstrates direct attribute access (`m.E_H`, `m.E_H.unit`, …).
• Runs a minuscule calculation so users see how `Quantity` cooperates
  with normal Python maths.
• Finds every reaction referencing this molecule and prints the most
  common use-cases (`deltaG0`, `computed_barrier` … with units).
"""
from __future__ import annotations

import sys
import json
import pathlib
import statistics
import h5py
import pandas as pd

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from molecode_utils.molecule import Molecule

# ----------------------------------------------------------------------
# CLI parsing (two optional args)
# ----------------------------------------------------------------------
h5_path = pathlib.Path("data/molecode-data-v0.1.0.h5")
mol_idx_filter = 99

print("=" * 72)
print(f"Opening HDF5: {h5_path.resolve()}\n")

with h5py.File(h5_path, "r") as h5:
    mol_ds = h5["molecules"]
    units_mol = mol_ds.attrs["column_units"]

    # --------------------------------------------------------------
    # pick the molecule record
    # --------------------------------------------------------------
    if mol_idx_filter is None:
        rec = mol_ds[0]                      # first entry
    else:
        mask = mol_ds["mol_idx"][:] == mol_idx_filter
        idx_arr = mask.nonzero()[0]
        if not idx_arr.size:
            raise SystemExit(f"mol_idx {mol_idx_filter} not found.")
        rec = mol_ds[idx_arr[0]]

    m = Molecule.from_row(rec, units_mol)

    # --------------------------------------------------------------
    # feature walkthrough
    # --------------------------------------------------------------
    print(f"▶ Molecule ID  : {m.id}")
    print(f"▶ SMILES       : {m.smiles} (unit = {m.smiles.unit})")
    print(f"▶ Dataset tag  : {m.dataset.value}")
    print()

    # show *all* numeric / descriptor columns in one neat table
    print("All molecule descriptors")
    print("-" * 34)
    for attr in sorted(
        a
        for a in dir(m)
        if (
            not a.startswith("_")          # skip dunders / privates
            and a not in {"id"}            # we printed id separately
            and not callable(getattr(m, a))
        )
    ):
        q = getattr(m, attr)
        print(f"{attr:30s}: {q.value!s:<20} [{q.unit}]")
    print("-" * 34, "\n")

    # --------------------------------------------------------------
    # quick calculation example
    # --------------------------------------------------------------
    print("Mini-demo calculation")
    delta_Grad = m.E_rad_deprot          # Quantity in V
    delta_Gox  = m.E_ox_0               # Quantity in V
    avg_redox  = (float(delta_Grad) + float(delta_Gox)) / 2  # plain float
    print(f"  (E_rad_deprot + E_ox_0)/2  =  {avg_redox:.3f}  V\n")

    # --------------------------------------------------------------
    # find linked reactions
    # --------------------------------------------------------------
    rxn_df = pd.DataFrame(h5["reactions"][:])
    units_rxn = json.loads(h5["reactions"].attrs["column_units"])

# outside the with-block (file is closed) ---------------------------
linked = rxn_df[(rxn_df["oxid_idx"] == m.id) | (rxn_df["subst_idx"] == m.id)]

print(f"▶ Reactions involving molecule-{m.id}: {len(linked)} found\n")

for _, row in linked.iterrows():
    dg0   = row["deltaG0"]
    dgb   = row["computed_barrier"]
    rxn   = row["rxn_idx"]
    print(f"  Reaction {rxn:4d}:  ΔG0 = {dg0:.2f} {units_rxn['deltaG0']},"
          f"  ΔG‡ = {dgb:.2f} {units_rxn['computed_barrier']}")

if len(linked):
    median_barrier = statistics.median(linked["computed_barrier"])
    print(f"\n  Median computed barrier across these reactions: "
          f"{median_barrier:.2f} {units_rxn['computed_barrier']}")
print("\nDone ✔")
print("=" * 72)

with h5py.File("data/molecode-data-v0.1.0.h5") as h5:
    m = Molecule.from_row(h5["molecules"][0], h5["molecules"].attrs["column_units"])

print(m.info())
ds_v = m.dataset        # ['Cumenes_para', 'Phenols_para_HAT', 'Toluens_para']
ds_u = m.dataset.unit   # '-'
for ds in m.dataset:
    print("•", ds)


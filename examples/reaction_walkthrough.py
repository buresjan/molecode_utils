#!/usr/bin/env python
"""
examples/show_reaction_features.py
==================================

Hands-on tour of *all* `Reaction` capabilities
----------------------------------------------
• Loads **one** reaction (`rxn_idx` given on the CLI, or the first in the file).
• Shows direct attribute access (`r.deltaG0`, `r.deltaG0.unit`, …).
• Demonstrates round-tripping of `Quantity` objects in calculations.
• Explores the linked oxidant/substrate `Molecule` objects
  (`r.oxidant.smiles`, `r.substrate.target_atom`, …).
• Prints a neat table of every reaction-level descriptor with its unit.
• Computes a couple of mini metrics so people see how smoothly everything
  behaves with normal Python / NumPy maths.
"""
from __future__ import annotations

import json
import pathlib
import statistics
import textwrap

import h5py
import pandas as pd

# Import helpers from the installed package
from molecode_utils.molecule import Molecule
from molecode_utils.reaction  import Reaction

h5_path        = pathlib.Path("data/molecode-data-v0.1.0.h5")
rxn_idx_filter = 9

print("=" * 80)
print(f"Opening HDF5: {h5_path.resolve()}\n")

with h5py.File(h5_path, "r") as h5:
    # ------------------------ load molecules -------------------------
    mol_ds     = h5["molecules"]
    units_mol  = mol_ds.attrs["column_units"]
    mol_df     = pd.DataFrame(mol_ds[:])

    # Build a *single* lookup dict once – all Reaction objects reuse it
    mol_lookup = {
        int(rec["mol_idx"]): Molecule.from_row(rec, units_mol)
        for _, rec in mol_df.iterrows()
    }

    # ------------------------ select reaction ------------------------
    rxn_ds     = h5["reactions"]
    import json
    units_rxn = json.loads(rxn_ds.attrs["column_units"])

    if rxn_idx_filter is None:
        rec = rxn_ds[0]             # first entry
    else:
        mask = rxn_ds["rxn_idx"][:] == rxn_idx_filter
        idx  = mask.nonzero()[0]
        if not idx.size:
            raise SystemExit(f"rxn_idx {rxn_idx_filter} not found.")
        rec = rxn_ds[idx[0]]

    r = Reaction.from_row(rec, units_rxn, molecule_lookup=mol_lookup)

# ----------------------------------------------------------------------
# Show the basics
# ----------------------------------------------------------------------
print(f"▶ Reaction ID     : {r.id}")
print(f"▶ Oxidant mol_idx : {r.oxidant.id}")
print(f"▶ Substrate mol_idx: {r.substrate.id}\n")

# quick peek at participants
print("Oxidant SMILES  :", r.oxidant.smiles.value)
print("Substrate SMILES:", r.substrate.smiles.value)
print()

# ----------------------------------------------------------------------
# Walk through *all* reaction-level descriptors
# ----------------------------------------------------------------------
print("All reaction descriptors")
print("-" * 60)
for attr in sorted(
    a for a in dir(r)
    if (
        not a.startswith("_")
        and a not in {"id", "oxidant", "substrate"}
        and not callable(getattr(r, a))
    )
):
    q = getattr(r, attr)
    print(f"{attr:28s}: {q.value!s:<18} [{q.unit}]")
print("-" * 60, "\n")

# ----------------------------------------------------------------------
# Mini calculation examples
# ----------------------------------------------------------------------
dg0   = r.deltaG0                 # Quantity (kcal/mol)
dgb   = r.computed_barrier        # Quantity (kcal/mol)
delt  = float(dgb) - float(dg0)   # plain float

print("Mini-demo calculations")
print(textwrap.dedent(f"""
    • ΔG0           = {dg0.value:.2f}  {dg0.unit}
    • ΔG‡ (barrier) = {dgb.value:.2f}  {dgb.unit}
    • ΔG‡ – ΔG0     = {delt:.2f}  {dgb.unit}
"""))

# Compare oxidant vs substrate *target_atom* just for fun
print("Target atoms")
print("  Oxidant   :", r.oxidant.target_atom.value)
print("  Substrate :", r.substrate.target_atom.value, "\n")

# ----------------------------------------------------------------------
# Scan the whole reactions table – find every reaction that involves
# this *same* oxidant or substrate molecule, show a quick statistic
# ----------------------------------------------------------------------
with h5py.File(h5_path, "r") as h5:
    rxn_df = pd.DataFrame(h5["reactions"][:])

linked = rxn_df[
    (rxn_df["oxid_idx"] == r.oxidant.id) |
    (rxn_df["subst_idx"] == r.substrate.id)
]

print(f"▶ Other reactions sharing either participant: {len(linked)} found")

if len(linked):
    median_barrier = statistics.median(linked["computed_barrier"])
    print(f"  Median computed barrier in that sub-set: "
          f"{median_barrier:.2f} {units_rxn['computed_barrier']}")
print("\nDone ✔")
print("=" * 80)

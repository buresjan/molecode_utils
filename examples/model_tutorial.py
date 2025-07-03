#!/usr/bin/env python
"""
examples/model_tutorial.py
==========================

Demonstration of the built‑in Marcus‑type models and how to create your
own custom model.
"""
from __future__ import annotations

import pathlib
import pandas as pd

from molecode_utils.dataset import Dataset
from molecode_utils.model import Model, ModelS, ModelM1, ModelM2, ModelM3, ModelM4

# ---------------------------------------------------------------------
# Custom model example
# ---------------------------------------------------------------------
class MyModel(Model):
    """Very naive linear model used purely for demonstration."""

    name = "DIY"

    def _predict_one(self, rxn):
        dg0 = float(getattr(rxn, "deltaG0", 0))
        w_r = float(getattr(rxn, "RC_formation_energy", 0))
        return 0.4 * dg0 + 0.2 * w_r


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
H5_PATH = pathlib.Path("data/molecode-data-v0.1.0.h5")

print("=" * 80)
print(f"Opening archive: {H5_PATH.resolve()}\n")

with Dataset.from_hdf(H5_PATH) as ds:
    phenols = ds.filter(datasets=["Phenols"])
    r_example = phenols[88]

    models = {
        "S": ModelS(lambda_000=20),
        "M1": ModelM1(),
        "M2": ModelM2(),
        "M3": ModelM3(),
        "M4": ModelM4(),
        "DIY": MyModel(),
    }

    print(f"Phenols subset size: {len(phenols)} reactions")
    print(f"Example reaction idx: {r_example.id}")
    print(
        f"Computed ΔG‡ = {r_example.computed_barrier.value:.2f} {r_example.computed_barrier.unit}\n"
    )

    for name, model in models.items():
        pred = model.predict(r_example)
        err = model.residual(r_example)
        print(f"{name:>3} → predicted = {pred:6.2f}  | residual = {err:+6.2f} kcal/mol")

    print("\nBulk evaluation on subset …")
    results = []
    for name, model in models.items():
        df = model.evaluate(phenols)
        results.append((name, df.attrs["MAE"], df.attrs["RMSE"]))
    summary = pd.DataFrame(results, columns=["model", "MAE", "RMSE"])
    print(summary.to_string(index=False, formatters={"MAE": "{:.2f}".format, "RMSE": "{:.2f}".format}))

print("\nTutorial complete ✓")
print("=" * 80)

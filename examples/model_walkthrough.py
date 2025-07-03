from pathlib import Path
import sys
import pathlib
import textwrap

import pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from molecode_utils.dataset import Dataset
from molecode_utils.model import (
    ModelS,
    ModelM1,
    ModelM2,
    ModelM3,
    ModelM4,
)

# ──────────────────────────────────────────────────────────────────────
# 0.  Configuration – where is the *.h5* file?
#     (adjust the relative path if necessary)
# ──────────────────────────────────────────────────────────────────────
H5_PATH = Path(__file__).resolve().parents[1] / "data" / "molecode-data-v0.1.0.h5"


# ──────────────────────────────────────────────────────────────────────
# 1.  Load an *entire* Dataset view – acts as an in‑memory lens onto
#     the HDF5 archive.  The file is kept open for the lifetime of the
#     Dataset and closed automatically when we exit the context‑manager.
# ──────────────────────────────────────────────────────────────────────
print("\n>>> 1. Opening archive …")
with Dataset.from_hdf(H5_PATH) as ds:
    print(f"Total reactions in archive: {len(ds):,}")

    # ------------------------------------------------------------------
    # 1a. Peek at the reaction‑level dataframe (all numeric columns)
    # ------------------------------------------------------------------
    descr = ds.describe()
    print("\nReaction ΔG‡ / ΔG0 summary (full dataset):\n")
    print(descr.loc[:, [c for c in descr.columns if c.startswith("deltaG") or c.startswith("computed_barrier")]])

    # ------------------------------------------------------------------
    # 1b.  Narrow the view: suppose we only care about the “Phenols”
    #      substrate family (dataset tag contains that word).
    # ------------------------------------------------------------------
    phenols = ds.filter(datasets=["Phenols"])
    print(f"\nPhenols subset size: {len(phenols)} reactions")

    # ------------------------------------------------------------------
    # 2.  Instantiate the various models we have implemented.
    # ------------------------------------------------------------------
    models = {
        "S" : ModelS(),
        "M1": ModelM1(),
        "M2": ModelM2(),
        "M3": ModelM3(),
        "M4": ModelM4(),
    }

    # ------------------------------------------------------------------
    # 2a. Predict the barrier for *one* reaction (pick the first).
    # ------------------------------------------------------------------
    r_example = phenols[88]            # reactions are fetched lazily
    print("\n>>> 2. Single‑reaction prediction example (rxn_idx ="
          f" {r_example.id})")
    print(f"Computed ΔG‡ = {r_example.computed_barrier.value:.2f} {r_example.computed_barrier.unit}")

    for name, model in models.items():
        pred = model.predict(r_example)
        err  = model.residual(r_example)
        print(f"{name:>3}  →  predicted = {pred:6.2f}  |  residual = {err:+6.2f} kcal/mol")

    # ------------------------------------------------------------------
    # 3.  Bulk prediction over an *entire* Dataset view – returns a
    #     pandas.Series that plays nicely with further analysis.
    # ------------------------------------------------------------------
    print("\n>>> 3. Bulk prediction on Phenols subset …")

    results = []
    for name, model in models.items():
        df = model.evaluate(phenols)   # DataFrame with .attrs[MAE/RMSE]
        mae  = df.attrs["MAE"]
        rmse = df.attrs["RMSE"]
        results.append((name, mae, rmse))

        # keep the predictions for later plotting / exploration
        phenols.reactions_df()[f"pred_{name}"] = df[f"{name}_pred"].values

    summary = pd.DataFrame(results, columns=["model", "MAE", "RMSE"])
    print("\nModel accuracy on Phenols subset (kcal mol⁻¹):\n")
    print(summary.to_string(index=False, formatters={"MAE": "{:.2f}".format, "RMSE": "{:.2f}".format}))

    # ------------------------------------------------------------------
    # 4.  Quick hint: because everything is a pandas object at this
    #     point, you are free to do any further analysis / plotting:
    #
    #     >>> import matplotlib.pyplot as plt
    #     >>> df = model.evaluate(ds)      # whole archive
    #     >>> df["residual"].hist(bins=40)
    # ------------------------------------------------------------------

print("\nWalkthrough complete ✔︎")

# =============================================================================
#  *What next?*
#  • Try different Dataset.filter() combinations – e.g.
#      ▸ ds.filter(query="deltaG0 < -5 and asynchronicity > 0")
#      ▸ ds.filter(KED_H__gt=0.8)
#  • Create scatter plots ΔG‡_computed vs ΔG‡_predicted.
#  • Drill down into the worst outliers by sorting on |residual|.
# =============================================================================

#!/usr/bin/env python
"""
examples/sequential_analysis.py
===============================

Demonstration of a simple sequential analysis workflow using
:meth:`molecode_utils.Dataset` helpers and :class:`ModelM4`.
The script loads the archive, performs a couple of filtering steps,
plots intermediate results and finally evaluates ``ModelM4``.
"""
from __future__ import annotations

import pathlib
import matplotlib.pyplot as plt
import pandas as pd

from molecode_utils.dataset import Dataset
from molecode_utils.model import ModelM4

# ---------------------------------------------------------------------
# Configuration – path to the MoleCode archive
# ---------------------------------------------------------------------
H5_PATH = pathlib.Path("data/molecode-data-v0.1.0.h5")
ASSET_DIR = pathlib.Path("examples/assets")
ASSET_DIR.mkdir(exist_ok=True)

print("=" * 80)
print(f"Loading archive: {H5_PATH.resolve()}\n")

with Dataset.from_hdf(H5_PATH) as ds:
    print(f"Total reactions in archive : {len(ds)}\n")

    # ── 1. full dataset overview ────────────────────────────────────
    df_all = ds.reactions_df()
    unit_sigma = ds[0].asynchronicity.unit
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.scatter(df_all["rxn_idx"], df_all["asynchronicity"], s=8, alpha=0.6)
    ax.set_xlabel("reaction index")
    ax.set_ylabel(f"asynchronicity [{unit_sigma}]")
    fig.tight_layout()
    out1 = ASSET_DIR / "async_vs_index.png"
    fig.savefig(out1)
    print(f"Figure saved: {out1.name}\n")

    # ── 2. filter on barrier + KED  ─────────────────────────────────
    ds_step1 = ds.filter(computed_barrier__gt=0, KED_react_atoms__gt=0.6)
    print(f"After barrier/KED filter   : {len(ds_step1)} reactions")
    df_step1 = ds_step1.reactions_df()
    unit_dg0 = ds_step1[0].deltaG0.unit
    unit_eta = ds_step1[0].frustration.unit
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.scatter(df_step1["deltaG0"], df_step1["frustration"], s=8, alpha=0.6)
    ax.set_xlabel(f"ΔG0 [{unit_dg0}]")
    ax.set_ylabel(f"frustration [{unit_eta}]")
    fig.tight_layout()
    out2 = ASSET_DIR / "dg0_vs_frustration.png"
    fig.savefig(out2)
    print(f"Figure saved: {out2.name}\n")

    # ── 3. oxidant E_H filter ───────────────────────────────────────
    ds_step2 = ds_step1.filter(func=lambda r: float(r.oxidant.E_H) > 1.0)
    print(f"With oxidant E_H > 1.0     : {len(ds_step2)} reactions\n")

    # ── 4. model evaluation ─────────────────────────────────────────
    print("Evaluating ModelM4 …")
    m4 = ModelM4()
    eval_df = m4.evaluate(ds_step2)
    mae = eval_df.attrs["MAE"]
    rmse = eval_df.attrs["RMSE"]
    print(f"MAE  = {mae:.2f} kcal/mol")
    print(f"RMSE = {rmse:.2f} kcal/mol\n")

    # join with dataset names for colouring
    plot_df = eval_df.join(
        ds_step2.reactions_df().set_index("rxn_idx")["datasets_str"],
        how="left",
    )
    codes, labels = pd.factorize(plot_df["datasets_str"])
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sc = ax.scatter(
        plot_df["actual"],
        plot_df["M4_pred"],
        c=codes,
        cmap="tab20",
        s=12,
        alpha=0.8,
    )
    lims = [plot_df[["actual", "M4_pred"]].min().min(),
            plot_df[["actual", "M4_pred"]].max().max()]
    ax.plot(lims, lims, ls="--", color="gray", lw=1)
    ax.set_xlabel(f"computed barrier [{ds_step2[0].computed_barrier.unit}]")
    ax.set_ylabel(f"ModelM4 predicted [{ds_step2[0].computed_barrier.unit}]")
    fig.tight_layout()
    out3 = ASSET_DIR / "modelM4_pred_vs_computed.png"
    fig.savefig(out3)
    print(f"Figure saved: {out3.name}\n")

print("Tutorial complete ✓")
print("=" * 80)

"""Dataset helpers for the MoleCode archive.

This module exposes two convenience classes:

``MolecodeArchive``
    Context manager around a single `.h5` file. The archive lazily caches
    :class:`Molecule` and :class:`Reaction` objects and can return the raw
    tables as :class:`pandas.DataFrame` instances.

``Dataset``
    A lightweight view onto a subset of reactions stored in a
    ``MolecodeArchive``. It provides pandas-like filtering and DataFrame
    export helpers.
"""
from __future__ import annotations

import contextlib
import json
import pathlib
import re
from typing import Callable, Dict, Iterable, List, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd

from .molecule import Molecule
from .reaction import Reaction


# ────────────────────────────────────────────────────────────────────
# 1. Low-level archive wrapper
# ────────────────────────────────────────────────────────────────────
class MolecodeArchive(contextlib.AbstractContextManager):
    """Wrapper around a MoleCode ``.h5`` archive with caching.

    Parameters
    ----------
    path : str or pathlib.Path
        Location of the HDF5 file.
    """

    def __init__(self, path: str | pathlib.Path) -> None:
        self._path = pathlib.Path(path)
        self._h5: h5py.File | None = None

        # lazy caches
        self._mol_cache: Dict[int, Molecule] = {}
        self._rxn_cache: Dict[int, Reaction] = {}

        # dataframe caches (cheap to build, but we keep them anyway)
        self._mol_df: pd.DataFrame | None = None
        self._rxn_df: pd.DataFrame | None = None

        # dataset-lookup helpers
        self._code_lookup: np.ndarray | None = None
        self._rxn_dataset_codes: np.ndarray | None = None

    # −− context-manager plumbing −−–––––––––––––––––––––––––––––––––––
    def __enter__(self) -> "MolecodeArchive":
        self._h5 = h5py.File(self._path, "r")

        # pre-load dataset-code helpers (used by Dataset.filter)
        self._code_lookup = np.array(
            self._h5["metadata/dataset_lookup"]
        ).astype(str)
        self._rxn_dataset_codes = self._h5["reactions_dataset_codes"][:]

        return self

    def __exit__(self, exc_type, exc, tb):
        if self._h5 is not None:
            self._h5.close()
        self._h5 = None

    # −− internal helpers −−–––––––––––––––––––––––––––––––––––––––––––
    @staticmethod
    def _load_units(ds: h5py.Dataset) -> dict[str, str]:
        raw = ds.attrs["column_units"]
        return json.loads(raw.decode() if isinstance(raw, (bytes, bytearray)) else raw)

    # −− public dataframe dumps (lazy) −−––––––––––––––––––––––––––––––
    def molecules_df(self) -> pd.DataFrame:
        """Return a copy of the molecules table.

        Returns
        -------
        pandas.DataFrame
            All molecule records stored in the archive.
        """
        if self._mol_df is None:
            self._mol_df = pd.DataFrame(self._h5["molecules"][:])
        # hand out a copy so callers can mutate without affecting the cache
        return self._mol_df.copy()

    def reactions_df(self, *, add_dataset_str: bool = True) -> pd.DataFrame:
        """
        Return a *copy* of the reactions table.

        Parameters
        ----------
        add_dataset_str
            If True, ensure the ``datasets_str`` column (comma-separated dataset
            names) is present.  The column will be injected lazily the first
            time it’s requested, even if the DataFrame had already been cached
            earlier without it.
        """
        # ––––– load & cache on first access –––––––––––––––––––––––––––
        if self._rxn_df is None:
            self._rxn_df = pd.DataFrame(self._h5["reactions"][:])

        # ––––– ensure extra column if requested –––––––––––––––––––––––
        want_col = add_dataset_str
        have_col = "datasets_str" in self._rxn_df.columns
        if want_col and not have_col:
            lookup = self._code_lookup
            self._rxn_df["datasets_str"] = [
                ",".join(lookup.take(arr)) for arr in self._rxn_dataset_codes
            ]

        # always hand out a copy so callers can mutate freely
        return self._rxn_df.copy()


    # −− cached object access −−–––––––––––––––––––––––––––––––––––––––
    def molecule(self, mol_idx: int) -> Molecule:
        if mol_idx in self._mol_cache:
            return self._mol_cache[mol_idx]

        ds     = self._h5["molecules"]
        mask   = ds["mol_idx"][:] == mol_idx
        idx    = mask.nonzero()[0]
        if not idx.size:
            raise KeyError(f"mol_idx {mol_idx} not found")

        row    = ds[idx[0]]
        units  = self._load_units(ds)
        m_obj  = Molecule.from_row(row, units)
        self._mol_cache[mol_idx] = m_obj
        return m_obj

    def reaction(self, rxn_idx: int) -> Reaction:
        if rxn_idx in self._rxn_cache:
            return self._rxn_cache[rxn_idx]

        ds     = self._h5["reactions"]
        mask   = ds["rxn_idx"][:] == rxn_idx
        idx    = mask.nonzero()[0]
        if not idx.size:
            raise KeyError(f"rxn_idx {rxn_idx} not found")

        row    = ds[idx[0]]
        units  = self._load_units(ds)

        ox_id, sub_id = int(row["oxid_idx"]), int(row["subst_idx"])
        mol_lookup: Mapping[int, Molecule] = {
            ox_id:  self.molecule(ox_id),
            sub_id: self.molecule(sub_id),
        }

        r_obj  = Reaction.from_row(row, units, molecule_lookup=mol_lookup)
        self._rxn_cache[rxn_idx] = r_obj
        return r_obj


# ────────────────────────────────────────────────────────────────────
# 2. Higher-level *view* class
# ────────────────────────────────────────────────────────────────────
class Dataset:
    """A *subset* of reactions inside one `MolecodeArchive`.

    Notes
    -----
    * Holds only the **rxn_idx set**; all heavy lifting delegated to the
      underlying archive object.
    * Chainable `.filter()` returns *new* Dataset instances (immutable view
      pattern à la pandas).
    """

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    def __init__(
        self,
        archive: MolecodeArchive,
        rxn_idx_list: Sequence[int] | None = None,
    ) -> None:
        self._arc = archive
        self._rxn_ids: List[int] = (
            list(rxn_idx_list) if rxn_idx_list is not None
            else archive.reactions_df(add_dataset_str=False)["rxn_idx"].tolist()
        )

        # cache for the reactions DataFrame with molecule columns joined in
        self._joined_df: pd.DataFrame | None = None

    @classmethod
    def from_hdf(cls, path: str | pathlib.Path) -> "Dataset":
        arc = MolecodeArchive(path)
        arc.__enter__()                       # user will close via Dataset.close()
        return cls(arc)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def close(self):
        """Close the underlying HDF5 file (good practice in scripts)."""
        self._arc.__exit__(None, None, None)

    # make it context-manager friendly too
    def __enter__(self):  # noqa: D401
        return self
    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ------------------------------------------------------------------
    # core container interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._rxn_ids)

    def __iter__(self):
        for idx in self._rxn_ids:
            yield self._arc.reaction(idx)

    def __getitem__(self, idx: int | slice):
        if isinstance(idx, slice):
            return Dataset(self._arc, self._rxn_ids[idx])
        return self._arc.reaction(self._rxn_ids[idx])

    # ------------------------------------------------------------------
    # CRUD-ish helpers
    # ------------------------------------------------------------------
    def add_reaction(self, reaction: Reaction) -> None:
        """Add an *existing* Reaction object to the view (no disk I/O)."""
        if reaction.id not in self._rxn_ids:
            self._rxn_ids.append(reaction.id)
            self._arc._rxn_cache[reaction.id] = reaction   # share cache

    def get_reaction(self, rxn_idx: int) -> Reaction:
        """Return Reaction or raise KeyError if not in *this* view."""
        if rxn_idx not in self._rxn_ids:
            raise KeyError(f"rxn_idx {rxn_idx} not in this Dataset view")
        return self._arc.reaction(rxn_idx)

    # ------------------------------------------------------------------
    # DataFrame exports
    # ------------------------------------------------------------------
    def reactions_df(self) -> pd.DataFrame:
        if self._joined_df is None:
            # reactions table with dataset names
            base = self._arc.reactions_df()
            base = base[base["rxn_idx"].isin(self._rxn_ids)]

            # bring in molecule-level columns for oxidant and substrate
            mol_df = self._arc.molecules_df()
            ox_cols = {
                col: f"oxidant.{col}" for col in mol_df.columns if col != "mol_idx"
            }
            ox_df = mol_df.rename(columns=ox_cols).rename(columns={"mol_idx": "oxid_idx"})

            sub_cols = {
                col: f"substrate.{col}" for col in mol_df.columns if col != "mol_idx"
            }
            sub_df = mol_df.rename(columns=sub_cols).rename(columns={"mol_idx": "subst_idx"})

            merged = base.merge(ox_df, on="oxid_idx", how="left").merge(sub_df, on="subst_idx", how="left")
            self._joined_df = merged.reset_index(drop=True)

        return self._joined_df.copy()

    def molecules_df(self) -> pd.DataFrame:
        """Return molecules referenced by reactions in this dataset.

        Returns
        -------
        pandas.DataFrame
            Molecule rows from the archive used by the view.
        """
        rxn_df = self.reactions_df()
        mol_ids = pd.unique(
            rxn_df[["oxid_idx", "subst_idx"]].values.ravel("K")
        )
        full = self._arc.molecules_df()
        # filter the archive table down to the molecules actually present here
        return full[full["mol_idx"].isin(mol_ids)].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Descriptive summary
    # ------------------------------------------------------------------
    def describe(self) -> pd.DataFrame:
        """pandas.describe() over numeric reaction columns."""
        df = self.reactions_df().select_dtypes(include=[np.number])
        return df.describe()

    # ------------------------------------------------------------------
    # Powerful, chainable filtering
    # ------------------------------------------------------------------
    def filter(
        self,
        *,
        query: str | None = None,
        func: Callable[[Reaction], bool] | None = None,
        datasets: Iterable[str] | None = None,
        **column_filters,
    ) -> "Dataset":
        """
        Return a new Dataset view with reactions that satisfy *all* criteria.

        Parameters
        ----------
        query
            pandas-style query string evaluated on the **reactions_df()**.
        func
            Arbitrary `lambda Reaction: bool` evaluated row-wise.
        datasets
            Iterable of dataset tag substrings (case-insensitive). Reaction
            kept if *any* tag matches its ``datasets_str`` column.
        **column_filters
            Column inequalities expressed as
            ``deltaG0__lt=-5``, ``KED_H__ge=0.7`` …
            Suffixes: __lt, __le, __gt, __ge, __eq, __ne.
        """
        # start with a mask that keeps all reactions
        mask = pd.Series(True, index=self._rxn_ids, dtype=bool)
        # pull the underlying DataFrame for evaluating the filters
        df = self.reactions_df().set_index("rxn_idx")

        # -- 1. pandas query ------------------------------------------------
        if query:
            # pandas-style expression evaluated on the reactions table
            mask &= df.eval(query, engine="python")

        # -- 2. dataset tag matching ---------------------------------------
        if datasets:
            # build a regex that ORs all dataset tags
            pat = "|".join(re.escape(tag) for tag in datasets)
            mask &= df["datasets_str"].str.contains(pat, case=False, regex=True)

        # -- 3. column inequality mini-dsl ---------------------------------
        op_map = {
            "__lt": "<",
            "__le": "<=",
            "__gt": ">",
            "__ge": ">=",
            "__eq": "==",
            "__ne": "!=",
        }
        for key, value in column_filters.items():
            for suffix, sym in op_map.items():
                if key.endswith(suffix):
                    col = key[: -len(suffix)]
                    expr_col = f"`{col}`" if not col.isidentifier() else col
                    expr = f"{expr_col} {sym} {repr(value)}"
                    # evaluate the inequality expression on the DataFrame
                    mask &= df.eval(expr, engine="python")
                    break
            else:
                raise ValueError(f"Bad filter key '{key}' – missing comparison suffix")

        # -- 4. lambda Reaction filter -------------------------------------
        if func:
            meets = []
            for rxn_idx in mask[mask].index:
                r = self._arc.reaction(int(rxn_idx))
                meets.append(func(r))
            mask &= pd.Series(meets, index=mask[mask].index)

        return Dataset(self._arc, mask[mask].index.tolist())

    # pretty repr
    def __repr__(self) -> str:        # pragma: no cover
        return f"<Dataset: {len(self)} reactions>"

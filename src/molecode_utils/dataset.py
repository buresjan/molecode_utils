"""
src/molecode_utils/dataset.py
=============================

Two complementary classes
-------------------------
1. MolecodeArchive
   • Very thin context-manager over one *.h5* file.
   • Caches Molecule / Reaction objects the first time they’re requested.
   • Gives dataframe dumps if you need full table access.

2. Dataset
   • A *view* onto **a subset of Reaction idxs** inside a MolecodeArchive.
   • Chainable `.filter()` based on:
       · pandas-style query strings,
       · free-form `lambda Reaction: bool`,
       · dataset tags (e.g. "Phenols_para"),
       · arbitrary column inequalities supplied as kwargs.
   • Easy `.reactions_df()` / `.molecules_df()` exports.
   • `add_reaction()`, `get_reaction()`, rich `__iter__` / `__len__`.
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
    """Light wrapper around a *.h5* MoleCode file with caching."""

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
        if self._mol_df is None:
            self._mol_df = pd.DataFrame(self._h5["molecules"][:])
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
        full = self._arc.reactions_df()            # already has datasets_str
        return full[full["rxn_idx"].isin(self._rxn_ids)].reset_index(drop=True)

    def molecules_df(self) -> pd.DataFrame:
        rxn_df = self.reactions_df()
        mol_ids = pd.unique(
            rxn_df[["oxid_idx", "subst_idx"]].values.ravel("K")
        )
        full = self._arc.molecules_df()
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
        mask = pd.Series(True, index=self._rxn_ids, dtype=bool)
        df   = self.reactions_df().set_index("rxn_idx")

        # -- 1. pandas query ------------------------------------------------
        if query:
            mask &= df.eval(query, engine="python")

        # -- 2. dataset tag matching ---------------------------------------
        if datasets:
            pat = "|".join(re.escape(tag) for tag in datasets)
            mask &= df["datasets_str"].str.contains(pat, case=False, regex=True)

        # -- 3. column inequality mini-dsl ---------------------------------
        op_map = {
            "__lt": "<", "__le": "<=", "__gt": ">", "__ge": ">=",
            "__eq": "==", "__ne": "!=",
        }
        for key, value in column_filters.items():
            for suffix, sym in op_map.items():
                if key.endswith(suffix):
                    col = key[: -len(suffix)]
                    expr = f"{repr(value)} {sym} {col}" if sym in ("<", "<=") else \
                           f"{col} {sym} {repr(value)}"
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

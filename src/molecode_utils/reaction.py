# src/molecode_utils/reaction.py
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from typing import Any, Mapping, Dict

# Re-use the helpers that already live in molecule.py
from .molecule import Quantity, UnitList, Molecule


class Reaction:
    """Immutable container for a reaction row.

    Parameters
    ----------
    id : int
        Primary key (column ``rxn_idx``).
    fields : Mapping[str, Quantity]
        Reaction level columns converted to :class:`Quantity` objects.
    oxidant : Molecule
        Instance referenced by ``oxid_idx``.
    substrate : Molecule
        Instance referenced by ``subst_idx``.

    Notes
    -----
    Any remaining reaction column becomes a dynamic attribute that returns a
    :class:`Quantity` instance.

    Examples
    --------
    >>> r.deltaG0, r.deltaG0.unit
    (-12.4, 'kcal/mol')
    >>> r.oxidant.smiles
    'O=C1OC(=O)C=CC1=O'
    """

    __slots__ = ("id", "oxidant", "substrate", "_fields", "_frozen")

    # ───────────────────────── construction ──────────────────────────
    def __init__(
        self,
        *,
        id: int,
        fields: Mapping[str, Quantity],
        oxidant: Molecule,
        substrate: Molecule,
    ) -> None:
        # allow setting attrs during __init__
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "oxidant", oxidant)
        object.__setattr__(self, "substrate", substrate)
        object.__setattr__(self, "_fields", dict(fields))
        object.__setattr__(self, "_frozen", True)   # lock immutability

    # ─────────────────── immutability enforcement ────────────────────
    def __setattr__(self, name: str, value: Any) -> None:          # pragma: no cover
        if getattr(self, "_frozen", False):
            raise AttributeError(f"{self.__class__.__name__} instances are immutable")
        object.__setattr__(self, name, value)

    # ───────────────────── dynamic attr access ───────────────────────
    def __getattr__(self, name: str) -> Quantity:                  # pragma: no cover
        try:
            return self._fields[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getitem__(self, key: str):
        return getattr(self, key)
    
    # tab-completion friendliness
    def __dir__(self):                                             # pragma: no cover
        base = set(super().__dir__())
        extras = {"oxidant", "substrate"}
        return sorted(base | extras | set(self._fields))

    def __repr__(self):
        dg0 = self._fields.get("deltaG0")
        dgb = self._fields.get("computed_barrier")
        extras = ""
        if dg0 and dgb:
            extras = f": ΔG0={dg0.value:.1f} {dg0.unit}, ΔG‡={dgb.value:.1f} {dgb.unit}"
        return f"Reaction<{self.id}>{extras}"

    # ─────────────────────── alternate ctor ─────────────────────────
    @classmethod
    def from_row(
        cls,
        row: Mapping[str, Any] | np.void,
        units: Mapping[str, str] | str | bytes,
        *,
        molecule_lookup: Mapping[int, Molecule],
    ) -> "Reaction":
        """Build a :class:`Reaction` from a structured HDF5 row.

        Parameters
        ----------
        row
            Mapping or raw ``np.void`` record from the ``reactions`` dataset.
        units
            Column → unit mapping or raw JSON bytes/str from dataset attrs.
        molecule_lookup
            Dict ``mol_idx → Molecule`` – the two keys referenced by
            *oxid_idx* and *subst_idx* **must** be present.
        """
        # --- normalise *units* -------------------------------------------
        if isinstance(units, (str, bytes)):
            units = json.loads(units)

        if isinstance(units, Mapping):
            units = {
                (k.decode() if isinstance(k, (bytes, bytearray)) else k): v
                for k, v in units.items()
            }

        # --- normalise *row* to a plain dict ------------------------------
        if isinstance(row, np.void):                         # raw scalar from h5py
            row = {name: row[name] for name in row.dtype.names}

        elif (
            hasattr(row, "__getitem__")                      # pandas single-row df
            and list(row.keys()) == [0]
            and isinstance(row[0], np.void)
        ):
            rec = row[0]
            row = {name: rec[name] for name in rec.dtype.names}

        # --- pull molecule ids first -------------------------------------
        ox_id   = int(row["oxid_idx"])
        sub_id  = int(row["subst_idx"])
        rxn_id  = int(row["rxn_idx"])

        try:
            oxidant   = molecule_lookup[ox_id]
            substrate = molecule_lookup[sub_id]
        except KeyError as exc:
            raise KeyError(
                f"Molecule idx {exc.args[0]} not found in *molecule_lookup*"
            ) from None

        # --- convert remaining columns into Quantity objects -------------
        field_objs: Dict[str, Quantity] = {}
        SKIP = {"rxn_idx", "oxid_idx", "subst_idx",  # already handled
                "oxid_smiles", "subst_smiles",       # redundant → go via Molecule
                "oxid_target_atom", "subst_target_atom",
                "oxid_target_atom_other_hs", "subst_target_atom_other_hs"}

        for col, val in row.items():
            if isinstance(col, (bytes, bytearray)):
                col = col.decode()

            if col in SKIP:
                continue

            # byte-string values → clean UTF-8
            if isinstance(val, (bytes, bytearray)):
                val = val.decode().rstrip()

            field_objs[col] = Quantity(val, units.get(col, "-"))

        return cls(id=rxn_id, fields=field_objs,
                   oxidant=oxidant, substrate=substrate)
    
    def unit(self, field: str) -> str:
        return getattr(self, field).unit

    def to_dict(self, *, include_units=False):
        if include_units:
            return {k: (q.value, q.unit) for k, q in self._fields.items()}
        return {k: q.value for k, q in self._fields.items()}

    def as_series(self, *, include_units=False):
        return pd.Series(self.to_dict(include_units=include_units))

    def info(self, *, width: int = 28) -> str:
        """Return a tabular dump of reaction-level fields.

        Parameters
        ----------
        width : int, optional
            Width of the name column, by default ``28``.

        Returns
        -------
        str
            Formatted table as a string.
        """
        lines = [
            f"Reaction id  = {self.id}",
            f"oxidant id   = {self.oxidant.id}",
            f"substrate id = {self.substrate.id}",
            "",
        ]
        for name in sorted(self._fields):
            q     = self._fields[name]
            value = q if not isinstance(q, (Quantity, UnitList)) else q.value
            unit  = getattr(q, "unit", "-")
            lines.append(f"{name:<{width}} : {value!s:<20} [{unit}]")
        return "\n".join(lines)

    def help(self) -> None:                                       # pragma: no cover
        """Print a short usage message for interactive sessions."""
        print("Available helpers:")
        print("  • r.info() – tabular dump of reaction-level fields")
        print("  • r.help() – this message")
        print("\nDynamic attributes include:")
        cols = ", ".join(sorted(self._fields)[:8]) + ", …"
        print(f"  {cols}")

    
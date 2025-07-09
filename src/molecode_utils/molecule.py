from __future__ import annotations

import json
import ast
import numpy as np
import pandas as pd
from typing import Any, Dict, Mapping


# ────────────────────────────────────────────────────────────────────────
# Helper: scalar value that remembers its unit
# ────────────────────────────────────────────────────────────────────────
class Quantity:
    """Scalar value with an attached physical unit."""

    __slots__ = ("_value", "unit")

    def __init__(self, value: Any, unit: str = "-") -> None:
        self._value = value
        self.unit = unit

    # ── representation ────────────────────────────────────────────────
    def __repr__(self) -> str:  # pragma: no cover
        return f"{self._value!r}"

    def __str__(self) -> str:  # pragma: no cover
        return str(self._value)

    # ── numeric conversions ───────────────────────────────────────────
    def __float__(self) -> float:  # pragma: no cover
        return float(self._value)

    def __int__(self) -> int:  # pragma: no cover
        return int(self._value)

    def __eq__(self, other: object) -> bool:  # pragma: no cover
        if isinstance(other, Quantity):
            return self._value == other._value and self.unit == other.unit
        return self._value == other

    # raw scalar if one really needs it
    @property
    def value(self) -> Any:
        """Return the underlying plain value (unitless)."""
        return self._value


class UnitList(list):
    """A list that also exposes `.unit` **and** a `.value` alias.

    Behaves like a regular list in every other respect.
    """

    __slots__ = ("unit",)

    def __init__(self, iterable, unit: str = "-"):
        super().__init__(iterable)
        self.unit = unit

    # keep old code that used `.value` from breaking
    @property
    def value(self):
        return self  # self *is* the data

    # friendlier representation in the REPL / logs
    def __repr__(self):
        return f"{list(self)!r}"  # prints the list only

    # optional: equality check includes the unit
    def __eq__(self, other):
        return (
            list.__eq__(self, other) and getattr(other, "unit", self.unit) == self.unit
        )


# ────────────────────────────────────────────────────────────────────────
# Main data class
# ────────────────────────────────────────────────────────────────────────
class Molecule:
    """Immutable container for a molecule row.

    Parameters
    ----------
    id : int
        Primary key (column ``mol_idx``).
    fields : Mapping[str, Quantity]
        Dictionary mapping column names to :class:`Quantity` instances.

    Notes
    -----
    Dynamic attribute access returns a :class:`Quantity` so you can write::

        m.E_H           # -> Quantity(-5.23, "V")
        m.E_H.unit      # -> "V"
        float(m.E_H)    # -> -5.23

    The aliases ``target_atom`` and ``target_atom_other_hs`` automatically pick
    the first non-null value from the substrate or oxidant specific columns.
    """

    __slots__ = ("id", "_fields", "_frozen")

    # ── construction ──────────────────────────────────────────────────
    def __init__(self, id: int, fields: Mapping[str, Quantity]) -> None:
        # during construction we may still set attributes
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "_fields", dict(fields))
        object.__setattr__(self, "_frozen", True)  # lock immutability

    # ── enforce immutability ──────────────────────────────────────────
    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover
        if getattr(self, "_frozen", False):
            raise AttributeError(f"{self.__class__.__name__} instances are immutable")
        object.__setattr__(self, name, value)

    # ── attribute access ──────────────────────────────────────────────
    def __getattr__(self, name: str) -> Quantity:
        try:
            return self._fields[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getitem__(self, key: str):
        return getattr(self, key)

    # tab-completion friendliness
    def __dir__(self):  # pragma: no cover
        return sorted(set(super().__dir__()) | set(self._fields))

    def __repr__(self) -> str:  # pragma: no cover
        smi = self._fields.get("smiles", Quantity("")).value
        return f"Molecule(id={self.id}, smiles={smi!r})"

    # ── alternate constructor ─────────────────────────────────────────
    @classmethod
    def from_row(
        cls,
        row: Mapping[str, Any],
        units: Mapping[str, str] | str | bytes,
    ) -> "Molecule":
        """Create a :class:`Molecule` from a pandas/NumPy record.

        Parameters
        ----------
        row
            Mapping column → value (e.g. pandas.Series or h5py record).
        units
            Column → unit mapping or raw JSON from ``h5py`` attributes.
        """
        # units might come as raw JSON bytes/str from h5py
        if isinstance(units, (str, bytes)):
            units = json.loads(units)

        if isinstance(units, Mapping):
            units = {
                (k.decode() if isinstance(k, (bytes, bytearray)) else k): v
                for k, v in units.items()
            }

        # ── NORMALISE *row* into a dict-like mapping ----------------------
        if isinstance(row, np.void):
            # Case 1 – raw structured scalar straight from h5py
            row = {name: row[name] for name in row.dtype.names}

        elif (
            isinstance(row, pd.DataFrame)
            and len(row) == 1
            and isinstance(row.iloc[0], np.void)
        ):
            rec = row.iloc[0]
            row = {name: rec[name] for name in rec.dtype.names}

        field_objs: Dict[str, Quantity] = {}

        # helper: rename the awkward “0” column
        def _rename(col: str) -> str:
            return "zero" if col == "0" else col

        # ── main loop over columns ────────────────────────────────────
        for col, val in row.items():
            # h5py gives bytes for column names → decode once
            if isinstance(col, (bytes, bytearray)):
                col = col.decode()
            name = _rename(col)

            if col == "dataset":
                # value is a string that looks like "['a', 'b', ...']"
                if isinstance(val, (bytes, bytearray)):
                    val = val.decode().rstrip()
                try:
                    parsed_list = ast.literal_eval(val)  # normal case
                    if not isinstance(parsed_list, list):  # sanity
                        raise ValueError
                except (SyntaxError, ValueError):
                    # Fallback: best-effort split of a truncated string
                    parsed_list = [
                        part.strip(" '\"")  # remove quotes/spaces
                        for part in val.strip(" []").split(",")
                        if part.strip()
                    ]

                field_objs[name] = UnitList(parsed_list, units.get(col, "-"))
            else:
                # byte-string values → clean UTF-8
                if isinstance(val, (bytes, bytearray)):
                    val = val.decode().rstrip()
                field_objs[name] = Quantity(val, units.get(col, "-"))

        # ── convenience aliases for the target atoms ──────────────────
        def _is_missing(v) -> bool:
            """Return True for '', None, or NaN."""
            if v is None:
                return True
            if isinstance(v, float) and np.isnan(v):
                return True
            if isinstance(v, str) and not v.strip():
                return True
            return False

        def _pick(first: str, second: str):
            """Return Quantity from the first non-missing field."""
            q1 = field_objs.get(first)
            if q1 is not None and not _is_missing(q1.value):
                return q1
            q2 = field_objs.get(second)
            if q2 is not None and not _is_missing(q2.value):
                return q2
            return None

        tgt = _pick("subst_target_atom", "oxid_target_atom")
        tgt_hs = _pick("subst_target_atom_other_hs", "oxid_target_atom_other_hs")

        if tgt is not None:
            field_objs["target_atom"] = tgt
        if tgt_hs is not None:
            field_objs["target_atom_other_hs"] = tgt_hs

        # ── pull out primary key, create instance ─────────────────────
        mol_idx_quant = field_objs.pop("mol_idx")
        return cls(id=mol_idx_quant.value, fields=field_objs)

    def unit(self, field: str) -> str:
        return getattr(self, field).unit

    def to_dict(self, *, include_units=False):
        if include_units:
            return {k: (q.value, q.unit) for k, q in self._fields.items()}

        def _clean(v):
            try:
                if isinstance(v, float) and np.isnan(v):
                    return None
            except Exception:
                pass
            return v

        return {k: _clean(q.value) for k, q in self._fields.items()}

    def as_series(self, *, include_units=False):
        return pd.Series(self.to_dict(include_units=include_units))

    def info(self, *, width: int = 28) -> str:
        """Return a nicely formatted summary of all data fields.

        Parameters
        ----------
        width : int, optional
            Width of the field name column, by default ``28``.

        Returns
        -------
        str
            Human readable table with values and units.
        """
        lines = [f"Molecule id = {self.id}", ""]
        for name in sorted(n for n in self._fields if not callable(self._fields[n])):
            q = self._fields[name]
            value = q if not isinstance(q, (Quantity, UnitList)) else q.value
            unit = getattr(q, "unit", "-")
            lines.append(f"{name:<{width}} : {value!s:<20} [{unit}]")
        return "\n".join(lines)

    def help(self) -> None:  # designed for REPL use
        """Print an on-the-fly cheat-sheet for Molecule methods."""
        print("Available helpers:")
        print("  • mol.info()   – tabular dump of all data fields, must be printed")
        print("  • mol.help()   – this message")
        print("\nDynamic attributes include:")
        print(f"  {cols}")

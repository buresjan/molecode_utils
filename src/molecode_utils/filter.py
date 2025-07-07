"""Convenient wrapper for :meth:`Dataset.filter` calls."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from .dataset import Dataset, Reaction


@dataclass
class Filter:
    """Reusable collection of Dataset filter criteria.

    Parameters
    ----------
    query
        pandas-style query string evaluated on the reactions dataframe.
    datasets
        Iterable of dataset tag substrings (case-insensitive).
    func
        Optional ``lambda Reaction: bool`` applied row-wise.
    reaction
        Mapping of reaction-level column inequalities.
    oxidant
        Mapping of oxidant (molecule-level) inequalities.
    substrate
        Mapping of substrate (molecule-level) inequalities.
    """

    query: str | None = None
    datasets: Iterable[str] | None = None
    func: Callable[[Reaction], bool] | None = None
    reaction: Mapping[str, Any] = field(default_factory=dict)
    oxidant: Mapping[str, Any] = field(default_factory=dict)
    substrate: Mapping[str, Any] = field(default_factory=dict)

    _op_map = {
        ">=" : "__ge",
        "<=" : "__le",
        ">"  : "__gt",
        "<"  : "__lt",
        "==" : "__eq",
        "="  : "__eq",
        "!=" : "__ne",
    }

    @staticmethod
    def _normalize(prefix: str, filters: Mapping[str, Any]) -> dict[str, Any]:
        """Translate shorthand operators to :meth:`Dataset.filter` keys."""
        out: dict[str, Any] = {}
        for raw_key, value in filters.items():
            key = raw_key.strip()
            # already expressed with __gt/__lt/... -> just prefix
            if "__" in key:
                col_key = key
            else:
                col_key = None
                # check for symbol operators
                for sym in sorted(Filter._op_map, key=len, reverse=True):
                    if key.endswith(sym):
                        base = key[: -len(sym)].strip()
                        col_key = f"{base}{Filter._op_map[sym]}"
                        break
                if col_key is None:
                    # default equality if no operator
                    base = key
                    col_key = f"{base}__eq"
            if prefix:
                col_key = f"{prefix}{col_key}"
            out[col_key] = value
        return out

    def apply(self, ds: Dataset) -> Dataset:
        """Return a new :class:`Dataset` filtered according to the criteria."""
        column_filters: dict[str, Any] = {}
        column_filters.update(self._normalize("", self.reaction))
        column_filters.update(self._normalize("oxidant.", self.oxidant))
        column_filters.update(self._normalize("substrate.", self.substrate))
        return ds.filter(
            query=self.query,
            func=self.func,
            datasets=self.datasets,
            **column_filters,
        )

    __call__ = apply

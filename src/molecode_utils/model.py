from __future__ import annotations

"""Model abstractions for predicting kinetic barriers.

The module defines an abstract :class:`Model` class and several Marcus-like
implementations (``ModelS`` and ``ModelM1``–``ModelM4``).  Models operate on
:class:`Reaction` objects and provide helpers for batch predictions and error
statistics.
"""

from abc import ABC, abstractmethod
from typing import Iterable, Mapping, Sequence, overload, Union

import math
import numbers
import numpy as np
import pandas as pd

from .molecule import Quantity
from .reaction import Reaction
from .dataset import Dataset

Number = Union[int, float, np.floating]

__all__ = [
    "Model",
    "ModelS",
    "ModelM1",
    "ModelM2",
    "ModelM3",
    "ModelM4",
]


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _q(value: Quantity | Number | None) -> float:
    """Return ``value`` as a bare float.

    Parameters
    ----------
    value : Quantity or Number or None
        Value that may carry units or be ``None``.

    Returns
    -------
    float
        ``value`` converted to ``float`` or ``NaN`` when missing.
    """
    if value is None:
        return math.nan
    if isinstance(value, Quantity):
        return float(value.value)
    if isinstance(value, numbers.Number):
        return float(value)
    return float(value)  # best‑effort fallback


# ──────────────────────────────────────────────────────────────────────
# Base abstraction
# ──────────────────────────────────────────────────────────────────────
class Model(ABC):
    """Abstract base‑class for any ΔG‡ predictor."""

    #: A short, human‑readable identifier (e.g. "M1")
    name: str = "Model"

    # ── compulsory low‑level primitive ────────────────────────────────
    @abstractmethod
    def _predict_one(self, rxn: Reaction) -> float:
        """Return barrier prediction **in kcal mol⁻¹** for *one* reaction."""

    # ── convenience public front‑end ──────────────────────────────────
    @overload
    def predict(self, item: Reaction) -> float: ...

    @overload
    def predict(self, item: Dataset | Iterable[Reaction]) -> pd.Series: ...

    def predict(self, item):  # type: ignore[override]
        """Predict barriers for ``item``.

        Parameters
        ----------
        item : Reaction or Dataset or Iterable[Reaction]
            Object(s) for which to compute predicted barriers.

        Returns
        -------
        float or pandas.Series
            A scalar for a single reaction or a series indexed by ``rxn_idx``
            when ``item`` is iterable.
        """
        if isinstance(item, Reaction):
            return self._predict_one(item)

        if isinstance(item, Dataset):
            iterator: Iterable[Reaction] = item  # Dataset is iterable
        else:  # assume already an iterable of Reaction objects
            iterator = item

        data = {rxn.id: self._predict_one(rxn) for rxn in iterator}
        return pd.Series(data, name=f"{self.name}_pred")

    # ── error / residual helpers ─────────────────────────────────────
    @overload
    def residual(self, item: Reaction) -> float: ...

    @overload
    def residual(self, item: Dataset | Iterable[Reaction]) -> pd.Series: ...

    def residual(self, item):  # type: ignore[override]
        """Difference between prediction and computed barrier.

        Parameters
        ----------
        item : Reaction or Dataset or Iterable[Reaction]
            Object(s) on which to compute residuals.

        Returns
        -------
        float or pandas.Series
            Residuals in the same shape as :meth:`predict` outputs.
        """
        if isinstance(item, Reaction):
            try:
                true = _q(item.computed_barrier)
            except AttributeError:
                raise ValueError("Reaction lacks *computed_barrier* attribute")
            return self._predict_one(item) - true

        # bulk case
        preds = self.predict(item)
        if isinstance(item, Dataset):
            true_series = pd.Series(
                {rxn.id: _q(rxn.computed_barrier) for rxn in item},
                name="computed_barrier",
            )
        else:  # iterable
            rxn_list = list(item)
            true_series = pd.Series(
                {rxn.id: _q(rxn.computed_barrier) for rxn in rxn_list},
                name="computed_barrier",
            )
        return preds - true_series

    # ── simple stats helper ──────────────────────────────────────────
    def evaluate(self, data: Dataset | Iterable[Reaction]) -> pd.DataFrame:
        """Compute predictions and error statistics for ``data``.

        Parameters
        ----------
        data : Dataset or Iterable[Reaction]
            Collection of reactions to evaluate.

        Returns
        -------
        pandas.DataFrame
            Data frame with columns ``pred``, ``actual`` and ``residual``.
            Mean absolute error and RMSE are stored in ``DataFrame.attrs``.
        """
        preds = self.predict(data)

        # obtain *actual* barriers
        actual = pd.Series(
            {rxn.id: _q(rxn.computed_barrier) for rxn in data}, name="actual"
        )
        resid = preds - actual

        df = pd.concat((preds, actual, resid.rename("residual")), axis=1)
        df.attrs["MAE"] = resid.abs().mean()
        df.attrs["RMSE"] = math.sqrt((resid**2).mean())
        return df

    # nicer str() / repr()
    # ----------------------------------------------------------------
    def __repr__(self):  # pragma: no cover
        return f"<{self.name}‑Model>"


# ──────────────────────────────────────────────────────────────────────
# Concrete Marcus / MCR‑style models
# ──────────────────────────────────────────────────────────────────────
class ModelS(Model):
    """*Model S* – self-reaction based Marcus expression.

    The reorganisation energy **λ₀₀₀** is *not* stored in the HDF archive;
    it is a **free, global parameter** of the model.  Supply it when
    instantiating the class – defaults to *0* (i.e. omitted) if you work
    with relative barriers where λ₀₀₀/4 cancels out.
    """

    name = "S"

    def __init__(self, lambda_000: float = 0.0):
        self.lambda_000 = float(lambda_000)

    # ------------------------------------------------------------------
    # Vectorised prediction on a DataFrame
    # ------------------------------------------------------------------
    def predict_df(self, df: pd.DataFrame) -> pd.Series:
        """Return model predictions for a reaction DataFrame."""
        lam = self.lambda_000
        w_R = df.get("RC_formation_energy")
        w_P = df.get("PC_formation_energy")
        sigma = df.get("asynchronicity")
        eta = df.get("frustration")
        dG0 = df.get("deltaG0")

        pred = (
            lam / 4.0 + 0.5 * (w_R + w_P) + 0.25 * (sigma.abs() - eta.abs()) + 0.5 * dG0
        )
        return pred.rename(f"{self.name}_pred")

    def _predict_one(self, rxn: Reaction) -> float:  # noqa: C901 (complex OK)
        # ── fetch building blocks ─────────────────────────────────–––
        lam = self.lambda_000  # ← user-supplied constant
        w_R = _q(getattr(rxn, "RC_formation_energy", None))
        w_P = _q(getattr(rxn, "PC_formation_energy", None))
        sigma = _q(getattr(rxn, "asynchronicity", None))
        eta = _q(getattr(rxn, "frustration", None))
        dG0 = _q(getattr(rxn, "deltaG0", None))

        # Marcus-style expression (kcal mol⁻¹)
        return (
            lam / 4.0 + 0.5 * (w_R + w_P) + 0.25 * (abs(sigma) - abs(eta)) + 0.5 * dG0
        )


class _MBase(Model):
    """Helper base‑class for all *M* family models (M1–M4)."""

    name = "M‑base"

    # ----------------------------------------------------------------
    # small auxiliaries
    # ----------------------------------------------------------------
    @staticmethod
    def _dGxx_yy(rxn: Reaction) -> float:
        """½(ΔG‡_XX + ΔG‡_YY)."""
        dG_xx = _q(getattr(rxn.oxidant, "self_exchange_barrier", None))
        dG_yy = _q(getattr(rxn.substrate, "self_exchange_barrier", None))
        return 0.5 * (dG_xx + dG_yy)

    @staticmethod
    def _wR_xx_yy(rxn: Reaction) -> float:
        """½(w_R,XX + w_R,YY) – uses *RC* formation energy of the two radicals."""
        wR_xx = _q(getattr(rxn.oxidant, "self_exchange_RC_formation", None))
        wR_yy = _q(getattr(rxn.substrate, "self_exchange_RC_formation", None))
        return 0.5 * (wR_xx + wR_yy)

    # ------------------------------------------------------------------
    # DataFrame helpers for vectorised predictions
    # ------------------------------------------------------------------
    @staticmethod
    def _dGxx_yy_df(df: pd.DataFrame) -> pd.Series:
        return 0.5 * (
            df["oxidant.self_exchange_barrier"] + df["substrate.self_exchange_barrier"]
        )

    @staticmethod
    def _wR_xx_yy_df(df: pd.DataFrame) -> pd.Series:
        return 0.5 * (
            df["oxidant.self_exchange_RC_formation"]
            + df["substrate.self_exchange_RC_formation"]
        )

    def _linear_term_df(self, df: pd.DataFrame) -> pd.Series:
        return self._dGxx_yy_df(df) + 0.5 * df["deltaG0"]

    # ----------------------------------------------------------------
    # building blocks reused across concrete subclasses
    # ----------------------------------------------------------------
    def _linear_term(self, rxn: Reaction) -> float:
        return self._dGxx_yy(rxn) + 0.5 * _q(rxn.deltaG0)


class ModelM1(_MBase):
    """M1  – *linear* self‑exchange based model."""

    name = "M1"

    def predict_df(self, df: pd.DataFrame) -> pd.Series:
        return self._linear_term_df(df).rename(f"{self.name}_pred")

    def _predict_one(self, rxn: Reaction) -> float:
        return self._linear_term(rxn)


class ModelM2(_MBase):
    """M2  – linear model + *formation‑energy* correction Δw."""

    name = "M2"

    def predict_df(self, df: pd.DataFrame) -> pd.Series:
        linear = self._linear_term_df(df)
        delta_w = 0.5 * (
            df["RC_formation_energy"] + df["PC_formation_energy"]
        ) - self._wR_xx_yy_df(df)
        return (linear + delta_w).rename(f"{self.name}_pred")

    def _predict_one(self, rxn: Reaction) -> float:
        linear = self._linear_term(rxn)
        w_R_XY = _q(getattr(rxn, "RC_formation_energy", None))
        w_P_XY = _q(getattr(rxn, "PC_formation_energy", None))
        delta_w = 0.5 * (w_R_XY + w_P_XY) - self._wR_xx_yy(rxn)
        return linear + delta_w


class ModelM3(_MBase):
    """M3  – adds the **quadratic Marcus term** f_q."""

    name = "M3"

    def predict_df(self, df: pd.DataFrame) -> pd.Series:
        linear = self._linear_term_df(df)
        denom = (
            df["oxidant.self_exchange_barrier"] + df["substrate.self_exchange_barrier"]
        )
        quad = 0.125 * (df["deltaG0"] ** 2) / denom
        quad = quad.where((denom != 0) & (~pd.isna(denom)))
        return (linear + quad).rename(f"{self.name}_pred")

    def _predict_one(self, rxn: Reaction) -> float:
        linear = self._linear_term(rxn)
        dG0 = _q(rxn.deltaG0)
        denom = _q(getattr(rxn.oxidant, "self_exchange_barrier", None)) + _q(
            getattr(rxn.substrate, "self_exchange_barrier", None)
        )
        if math.isnan(denom) or denom == 0:
            quad = math.nan
        else:
            quad = 0.125 * (dG0**2) / denom
        return linear + quad


class ModelM4(_MBase):
    """M4  – linear + Δw + *quadratic w‑corrected* term f_qw."""

    name = "M4"

    def _predict_one(self, rxn: Reaction) -> float:  # noqa: C901 (complexity OK)
        linear = self._linear_term(rxn)

        # Δw – formation energy correction ----------------------------
        w_R_XY = _q(getattr(rxn, "RC_formation_energy", None))
        w_P_XY = _q(getattr(rxn, "PC_formation_energy", None))
        delta_w = 0.5 * (w_R_XY + w_P_XY) - self._wR_xx_yy(rxn)

        # f_qw – quadratic term with *formation* correction -----------
        num = dG0_corr = _q(rxn.deltaG0) + w_P_XY - w_R_XY

        denom = (
            _q(getattr(rxn.oxidant, "self_exchange_barrier", None))
            + _q(getattr(rxn.substrate, "self_exchange_barrier", None))
            - self._wR_xx_yy(rxn) * 2  # because _wR_xx_yy already has the ½ factor
        )
        if math.isnan(denom) or denom == 0:
            quad = math.nan
        else:
            quad = 0.125 * (num**2) / denom

        return linear + delta_w + quad

    def predict_df(self, df: pd.DataFrame) -> pd.Series:
        linear = self._linear_term_df(df)
        w_R_XY = df["RC_formation_energy"]
        w_P_XY = df["PC_formation_energy"]
        delta_w = 0.5 * (w_R_XY + w_P_XY) - self._wR_xx_yy_df(df)

        num = df["deltaG0"] + w_P_XY - w_R_XY
        denom = (
            df["oxidant.self_exchange_barrier"]
            + df["substrate.self_exchange_barrier"]
            - 2 * self._wR_xx_yy_df(df)
        )
        quad = 0.125 * (num**2) / denom
        quad = quad.where((denom != 0) & (~pd.isna(denom)))

        return (linear + delta_w + quad).rename(f"{self.name}_pred")

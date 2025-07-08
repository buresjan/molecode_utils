from __future__ import annotations

"""Simple Plotly-backed figure helpers."""

from typing import Optional

import pandas as pd
import plotly.express as px

from .dataset import Dataset
from .model import Model
from .var_metadata import variable_metadata


class TwoDRxn:
    """Basic 2D scatter figure for reaction-level data."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        x: str,
        y: str,
        model: Optional[Model] = None,
        color_by: Optional[str] = None,
        group_by: Optional[str] = None,
        title: Optional[str] = None,
        latex_labels: bool = True,
    ) -> None:
        self.dataset = dataset
        self.x = x
        self.y = y
        self.model = model
        self.color_by = color_by
        self.group_by = group_by
        self.latex_labels = latex_labels

        df = dataset.reactions_df()
        self._decode_strings(df)
        if model is not None:
            df[f"{model.name}_pred"] = model.predict(dataset)
            df[f"{model.name}_resid"] = model.residual(dataset)
        self._df = df

        hover_cols = {"rxn_idx": True}
        if "oxidant.smiles" in df.columns:
            hover_cols["oxidant.smiles"] = True
        if "substrate.smiles" in df.columns:
            hover_cols["substrate.smiles"] = True

        self.title = title or self._make_title(x, y)
        labels = {
            x: self._make_label(x, latex=self.latex_labels),
            y: self._make_label(y, latex=self.latex_labels),
        }
        color_col = color_by or group_by
        self.figure = px.scatter(
            df,
            x=x,
            y=y,
            color=color_col,
            labels=labels,
            title=self.title,
            hover_data=hover_cols,
            template="plotly_white",
            height=700,
        )
        self.figure.update_layout(
            hovermode="closest",
            hoverlabel=dict(bgcolor="white"),
            xaxis=dict(showline=True, mirror=True, linecolor="black"),
            yaxis=dict(showline=True, mirror=True, linecolor="black"),
        )
        self.figure.update_layout(
            xaxis_title=self._make_label(x, latex=self.latex_labels),
            yaxis_title=self._make_label(y, latex=self.latex_labels),
        )
        self.figure.update_traces(marker=dict(size=6))

    @staticmethod
    def _decode_strings(df: pd.DataFrame) -> pd.DataFrame:
        """Decode byte-string columns in-place."""
        for col in df.select_dtypes(include=[object]).columns:
            df[col] = df[col].apply(
                lambda v: (
                    v.decode().rstrip() if isinstance(v, (bytes, bytearray)) else v
                )
            )
        return df

    @staticmethod
    def _make_label(var: str, *, latex: bool = True) -> str:
        meta = variable_metadata.get(var, {})
        if latex:
            latex_lbl = meta.get("latex", var)
            unit = meta.get("unit_latex", "")

            def strip_math(s: str) -> str:
                return s[1:-1] if s.startswith("$") and s.endswith("$") else s

            latex_lbl = strip_math(latex_lbl)
            unit = strip_math(unit)
            if unit:
                return f"${latex_lbl}\\;{unit}$"
            return f"${latex_lbl}$"

        # plain text label
        name = meta.get("name", var)
        unit_name = meta.get("unit_name", "")
        if unit_name:
            return f"{name} [{unit_name}]"
        return name

    @staticmethod
    def _make_title(x: str, y: str) -> str:
        mx = variable_metadata.get(x, {}).get("name", x)
        my = variable_metadata.get(y, {}).get("name", y)
        return f"{my} vs {mx}"

    def show(self):
        """Display the figure (shortcut for ``self.figure.show()``)."""
        self.figure.show()

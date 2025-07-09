from __future__ import annotations

"""Simple Plotly-backed figure helpers."""

from typing import Optional
import ast

import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

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
        fast_predict: bool = True,
        backend: str = "plotly",
    ) -> None:
        self.dataset = dataset
        self.x = x
        self.y = y
        self.model = model
        self.color_by = color_by
        self.group_by = group_by
        self.latex_labels = latex_labels
        self.backend = backend

        need_dataset_main = color_by == "dataset_main" or group_by == "dataset_main"
        df = dataset.reactions_df(add_dataset_main=need_dataset_main)
        self._decode_strings(df)
        if model is not None:
            if fast_predict and hasattr(model, "predict_df"):
                df[f"{model.name}_pred"] = model.predict_df(df)
                if "computed_barrier" in df.columns:
                    df[f"{model.name}_resid"] = (
                        df[f"{model.name}_pred"] - df["computed_barrier"]
                    )
            else:
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
        if self.backend == "plotly":
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
        elif self.backend == "matplotlib":
            self.figure = self._scatter_matplotlib(df, x, y, color_col, labels)
        else:
            raise ValueError("backend must be 'plotly' or 'matplotlib'")

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

    def _scatter_matplotlib(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        color_col: Optional[str],
        labels: dict[str, str],
    ):
        """Build a matplotlib scatter plot."""
        if self.group_by:
            groups = list(df[self.group_by].unique())
            fig, axes = plt.subplots(
                1,
                len(groups),
                figsize=(5 * len(groups), 5),
                sharex=True,
                sharey=True,
            )
            if len(groups) == 1:
                axes = [axes]
            for ax, grp in zip(axes, groups):
                sub = df[df[self.group_by] == grp]
                sc = ax.scatter(
                    sub[x],
                    sub[y],
                    c=sub[color_col] if color_col else None,
                    cmap="viridis",
                    s=20,
                )
                ax.set_title(str(grp))
                ax.set_xlabel(labels[x])
                ax.set_ylabel(labels[y])
            if color_col:
                fig.colorbar(sc, ax=axes)
        else:
            fig, ax = plt.subplots(figsize=(7, 5))
            sc = ax.scatter(
                df[x],
                df[y],
                c=df[color_col] if color_col else None,
                cmap="viridis",
                s=20,
            )
            ax.set_xlabel(labels[x])
            ax.set_ylabel(labels[y])
            if color_col:
                fig.colorbar(sc, ax=ax)
        fig.suptitle(self.title)
        return fig

    def show(self) -> None:
        """Display the figure (shortcut for ``self.figure.show()``)."""
        self.figure.show()


class TwoDMol:
    """Basic 2D scatter figure for molecule-level data."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        x: str,
        y: str,
        color_by: Optional[str] = None,
        group_by: Optional[str] = None,
        title: Optional[str] = None,
        latex_labels: bool = True,
        backend: str = "plotly",
    ) -> None:
        self.dataset = dataset
        self.x = x
        self.y = y
        self.color_by = color_by
        self.group_by = group_by
        self.latex_labels = latex_labels
        self.backend = backend

        df = dataset.molecules_df()
        TwoDRxn._decode_strings(df)

        if color_by == "dataset_main" or group_by == "dataset_main":
            df["dataset_main"] = df["dataset"].apply(self._extract_dataset_main)

        self._df = df

        self.title = title or TwoDRxn._make_title(x, y)
        labels = {
            x: TwoDRxn._make_label(x, latex=self.latex_labels),
            y: TwoDRxn._make_label(y, latex=self.latex_labels),
        }
        color_col = color_by or group_by
        if self.backend == "plotly":
            self.figure = px.scatter(
                df,
                x=x,
                y=y,
                color=color_col,
                labels=labels,
                title=self.title,
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
                xaxis_title=TwoDRxn._make_label(x, latex=self.latex_labels),
                yaxis_title=TwoDRxn._make_label(y, latex=self.latex_labels),
            )
            self.figure.update_traces(marker=dict(size=6))
        elif self.backend == "matplotlib":
            self.figure = TwoDRxn._scatter_matplotlib(self, df, x, y, color_col, labels)
        else:
            raise ValueError("backend must be 'plotly' or 'matplotlib'")

    @staticmethod
    def _extract_dataset_main(val: object) -> str:
        """Return the first dataset name from a list or string representation."""
        if not val:
            return ""
        if isinstance(val, list):
            return str(val[0])
        text = val.decode() if isinstance(val, (bytes, bytearray)) else str(val)
        text = text.lstrip(" [\"'")
        first = text.split(",", 1)[0]
        return first.strip().strip("'\"")

    def show(self) -> None:
        """Display the figure (shortcut for ``self.figure.show()``)."""
        self.figure.show()

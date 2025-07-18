import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from molecode_utils.dataset import Dataset
from molecode_utils.filter import Filter
from molecode_utils.figures import (
    TwoDRxn,
    TwoDMol,
    ThreeDRxn,
    ThreeDMol,
    Histogram,
)
from molecode_utils.model import ModelM1, ModelM2, ModelM3, ModelM4

# -----------------------------------------------------------------------------
# Dataset setup
# -----------------------------------------------------------------------------
DATA_PATH = "data/molecode-data-v0.1.0.h5"
full_ds = Dataset.from_hdf(DATA_PATH)

reaction_df = full_ds.reactions_df()
num_cols = reaction_df.select_dtypes(include="number").columns

# Numeric columns available in the molecules table
mol_df = full_ds.molecules_df()
mol_num_cols = mol_df.select_dtypes(include="number").columns

# Columns whose numeric ranges are finite (no NaN at either end)
finite_cols = [
    col
    for col in num_cols
    if pd.notna(reaction_df[col].min()) and pd.notna(reaction_df[col].max())
]

safe_col_ids = {col: col.replace(".", "_") for col in num_cols}
finite_ids = [safe_col_ids[c] for c in finite_cols]

all_tags = sorted(
    {
        tag.strip()
        for entry in reaction_df["datasets_str"]
        for tag in str(entry).split(",")
    }
)

# Pre-compute ranges for all numeric columns so we can build generic
# filtering inputs. Each pair spans the actual data range.
num_ranges = {col: (reaction_df[col].min(), reaction_df[col].max()) for col in num_cols}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
MODEL_OPTIONS = {
    "M1": ModelM1(),
    "M2": ModelM2(),
    "M3": ModelM3(),
    "M4": ModelM4(),
}


def _models_overview_table() -> html.Table:
    """Return static table of available models and their equations."""
    header = html.Thead(html.Tr([html.Th("Model"), html.Th("Equation")]))
    rows = [
        html.Tr([html.Td(m.name), html.Td(m.equation)]) for m in MODEL_OPTIONS.values()
    ]
    body = html.Tbody(rows)
    return html.Table([header, body], className="table table-sm")


def _model_stats_table(ds: Dataset) -> html.Table:
    """Return table with simple stats for each model on ``ds``."""
    rows = []
    df_all = ds.reactions_df()
    actual = df_all.get("computed_barrier")
    n_total = len(df_all)
    for m in MODEL_OPTIONS.values():
        eval_df = m.evaluate(ds)
        pred = eval_df[f"{m.name}_pred"]
        valid = pred.notna() & actual.notna()
        coverage = valid.sum() / n_total * 100 if n_total else 0.0
        if valid.sum() > 1:
            res = pred[valid] - actual[valid]
            mae = float(res.abs().mean())
            rmse = float((res**2).mean() ** 0.5)
            slope, intercept = np.polyfit(pred[valid], actual[valid], 1)
            mean_y = actual[valid].mean()
            ss_res = ((actual[valid] - pred[valid]) ** 2).sum()
            ss_tot = ((actual[valid] - mean_y) ** 2).sum()
            r2 = float("nan") if ss_tot == 0 else float(1 - ss_res / ss_tot)
            line = f"{slope:.2f}x + {intercept:.2f}"
        else:
            mae = rmse = r2 = float("nan")
            line = "–"
        rows.append(
            html.Tr(
                [
                    html.Td(m.name),
                    html.Td(f"{r2:.2f}"),
                    html.Td(f"{mae:.2f}"),
                    html.Td(f"{rmse:.2f}"),
                    html.Td(line),
                    html.Td(f"{coverage:.1f}%"),
                ]
            )
        )
    header = html.Thead(
        html.Tr(
            [
                html.Th("Model"),
                html.Th("R²"),
                html.Th("MAE"),
                html.Th("s"),
                html.Th("y ="),
                html.Th("Valid %"),
            ]
        )
    )
    body = html.Tbody(rows)
    return html.Table([header, body], className="table table-sm")


def dropdown(id_, options, value=None, multi=False, *, style=None):
    """Return a reusable ``dcc.Dropdown`` element."""

    fmt_opts = [o if isinstance(o, dict) else {"label": o, "value": o} for o in options]

    if style is None:
        style = {"width": "100%"}

    return dcc.Dropdown(
        options=fmt_opts,
        value=value,
        id=id_,
        multi=multi,
        clearable=False,
        style=style,
    )


# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------
external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css",
]
app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,
)
app.title = "Molecode Dashboard"


def _filter_tile():
    inputs = []
    for col in finite_cols:
        cid = safe_col_ids[col]
        tip_id = f"tip-{cid}"
        inputs.append(
            html.Div(
                [
                    html.Label(
                        [
                            f"{col} ",
                            html.I(
                                className="fa fa-circle-info text-secondary",
                                id=tip_id,
                                style={
                                    "cursor": "pointer",
                                    "marginLeft": "4px",
                                    "fontSize": "0.85rem",
                                },
                            ),
                        ],
                        className="mb-1",
                    ),
                    dbc.Tooltip(
                        f"Allowed range: {num_ranges[col][0]} – {num_ranges[col][1]}",
                        target=tip_id,
                        placement="right",
                    ),
                    dbc.InputGroup(
                        [
                            dcc.Input(
                                id=f"min-{cid}",
                                type="number",
                                value=num_ranges[col][0],
                                className="form-control",
                            ),
                            dbc.InputGroupText("–"),
                            dcc.Input(
                                id=f"max-{cid}",
                                type="number",
                                value=num_ranges[col][1],
                                className="form-control",
                            ),
                        ],
                        className="mb-2 w-100",
                    ),
                ],
                className="filter-row",
            )
        )

    content = html.Div(
        [
            html.Div(
                [
                    html.H5("Dataset Filtering", className="m-0"),
                    html.Div(
                        [
                            html.Button(
                                "Apply",
                                id="apply-filter-btn",
                                className="btn btn-primary btn-sm",
                            ),
                            html.Button(
                                "Clear",
                                id="clear-filter-btn",
                                className="btn btn-primary btn-sm ms-2",
                            ),
                        ],
                        style={
                            "display": "flex",
                            "gap": "6px",
                        },  # Optional: space between buttons
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                    "position": "sticky",
                    "top": 0,
                    "zIndex": 1,
                    "background": "white",
                    "paddingBottom": "4px",
                },
            ),
            html.Div(
                [
                    html.Label("Datasets", className="m-0"),
                    html.Button(
                        "Select All",
                        id="select-datasets-btn",
                        className="btn btn-outline-secondary btn-sm ms-2",
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "4px"},
            ),
            dropdown("dataset-dropdown", all_tags, multi=True),
            html.Div(inputs, style={"display": "grid", "rowGap": "6px"}),
        ],
        style={
            "border": "1px solid #ccc",
            "boxShadow": "0 2px 4px rgba(0,0,0,.04)",
            "padding": "10px",
            "overflowY": "auto",
            "minHeight": "0",
            "maxHeight": "100%",
        },
    )
    return content


def _figure_panel(idx: int) -> html.Div:
    """Return a single figure panel for the dashboard."""

    figure_type_dd = dropdown(
        {"type": "figtype", "pane": idx},
        ["TwoDRxn", "TwoDMol", "ThreeDRxn", "ThreeDMol", "Histogram"],
        value="TwoDRxn",
        style={"width": "100%"},
    )

    opts_btn = dbc.Button(
        "\u2699",
        id={"type": "opts-btn", "pane": idx},
        size="sm",
        className="mb-1",
        color="light",  # Use Bootstrap's "light" button for white/gray
        style={
            "backgroundColor": "white",
            "border": "1px solid #ccc",
        },  # optional refinement
    )

    header = html.Div(
        [figure_type_dd, opts_btn],
        style={"display": "flex", "alignItems": "center", "gap": "4px"},
    )
    graph = dcc.Graph(
        id={"type": "fig", "pane": idx},
        style={"flex": 1, "width": "100%", "height": "100%"},
        config={"responsive": True},
    )
    controls = dbc.Collapse(
        html.Div(
            id={"type": "controls", "pane": idx},
            style={"maxHeight": "160px", "overflowY": "auto"},
        ),
        id={"type": "collapse", "pane": idx},
        is_open=False,
    )
    return html.Div(
        [header, controls, graph],
        style={
            "display": "flex",
            "flexDirection": "column",
            "minHeight": 0,
            "minWidth": 0,
        },
    )


def _figure_board() -> html.Div:
    """Container holding the 2×2 grid of figure panels."""

    return html.Div(
        [_figure_panel(i + 1) for i in range(4)],
        style={
            "display": "grid",
            "gridTemplateColumns": "1fr 1fr",
            "gridTemplateRows": "1fr 1fr",
            "gap": "10px",
            "height": "100%",
            "padding": "8px",
            "border": "1px solid #ccc",
            "boxShadow": "0 2px 4px rgba(0,0,0,.04)",
        },
    )


def _model(name: str | None):
    """Return a model instance for the given name, or ``None``."""

    if not name or name == "None":
        return None
    return MODEL_OPTIONS.get(name)


from typing import Any


def _make_controls(fig_type: str, *, pane: int) -> list[Any]:
    """Return the controls appropriate for a figure type."""

    B = lambda role: {"type": "ctrl", "pane": pane, "role": role}
    DD = lambda role, opts, val: dropdown(B(role), opts, value=val)

    if fig_type == "TwoDRxn":
        color_opts = [
            "None",
            {"label": "Model Residual", "value": "Model Residual", "disabled": True},
            *num_cols,
        ]
        return [
            html.Label("X"),
            DD("x", num_cols, "deltaG0"),
            html.Label("Y"),
            DD("y", num_cols, "computed_barrier"),
            html.Label("Color"),
            DD("color", color_opts, "None"),
            html.Label("Model"),
            DD("model", ["None"] + list(MODEL_OPTIONS), "None"),
        ]

    if fig_type == "TwoDMol":
        color_opts = ["None", *mol_num_cols]
        return [
            html.Label("X"),
            DD("x", mol_num_cols, "pKaRH"),
            html.Label("Y"),
            DD("y", mol_num_cols, "E_H"),
            html.Label("Color"),
            DD("color", color_opts, "None"),
        ]

    if fig_type == "ThreeDRxn":
        color_opts = [
            "None",
            {"label": "Model Residual", "value": "Model Residual", "disabled": True},
            *num_cols,
        ]
        return [
            html.Label("X"),
            DD("x", num_cols, "deltaG0"),
            html.Label("Y"),
            DD("y", num_cols, "asynchronicity"),
            html.Label("Z"),
            DD("z", num_cols, "computed_barrier"),
            html.Label("Color"),
            DD("color", color_opts, "None"),
            html.Label("Model"),
            DD("model", ["None"] + list(MODEL_OPTIONS), "None"),
        ]

    if fig_type == "ThreeDMol":
        color_opts = ["None", *mol_num_cols]
        return [
            html.Label("X"),
            DD("x", mol_num_cols, "pKaRH"),
            html.Label("Y"),
            DD("y", mol_num_cols, "E_H"),
            html.Label("Z"),
            DD("z", mol_num_cols, "omega"),
            html.Label("Color"),
            DD("color", color_opts, "None"),
        ]

    return [
        html.Label("Variable"),
        DD("col", num_cols, "deltaG0"),
        html.Label("Bins"),
        dcc.Input(id=B("bins"), type="number", value=50, className="form-control"),
        html.Label("Color"),
        DD("color", ["None", "dataset_main"] + list(num_cols), "None"),
    ]


@app.callback(
    Output({"type": "controls", "pane": MATCH}, "children"),
    Input({"type": "figtype", "pane": MATCH}, "value"),
)
def render_controls(fig_type: str):
    ctx = dash.callback_context
    if not ctx.inputs_list:
        return dash.no_update
    pane = ctx.inputs_list[0]["id"].get("pane")
    return _make_controls(fig_type, pane=pane)


def _info_tile():
    content = html.Div(
        [
            dcc.Tabs(
                [
                    dcc.Tab(html.Div(id="dataset-info"), label="Dataset Info"),
                    dcc.Tab(
                        html.Div("Molecule lookup coming soon"), label="Molecule Lookup"
                    ),
                    dcc.Tab(
                        html.Div("Reaction lookup coming soon"), label="Reaction Lookup"
                    ),
                    dcc.Tab(
                        html.Div(_models_overview_table(), id="models-table"),
                        label="Models",
                    ),
                    dcc.Tab(html.Div(id="model-stats"), label="Model Stats"),
                ]
            )
        ],
        style={
            "border": "1px solid #ccc",
            "boxShadow": "0 2px 4px rgba(0,0,0,.04)",
            "padding": "10px",
        },
    )
    return content


app.layout = html.Div(
    [
        dcc.Store(id="filtered-indexes"),
        html.Div(
            [_filter_tile(), _info_tile()],
            style={
                "flex": "0 0 30%",
                "height": "100%",
                "minWidth": 0,
                "display": "grid",
                "gridTemplateRows": "2fr 2fr",
                "gap": "10px",
            },
        ),
        html.Div(
            [_figure_board()],
            style={
                "flex": "0 0 70%",
                "height": "100%",
                "minWidth": 0,
            },
        ),
    ],
    style={"display": "flex", "height": "100vh", "gap": "10px", "padding": "10px"},
)


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------
filter_states = []
for cid in finite_ids:
    filter_states.extend(
        [
            State(f"min-{cid}", "value"),
            State(f"max-{cid}", "value"),
        ]
    )


@app.callback(
    [
        Output("filtered-indexes", "data"),
        Output("dataset-info", "children"),
        Output("model-stats", "children"),
    ],
    Input("apply-filter-btn", "n_clicks"),
    State("dataset-dropdown", "value"),
    *filter_states,
)
def update_filters(n_clicks, datasets, *values):
    if n_clicks is None:
        filtered = full_ds
    else:
        flt = Filter()
        if datasets:
            flt.datasets = datasets
        it = iter(values)
        for col in finite_cols:
            min_v, max_v = next(it), next(it)

            if (min_v, max_v) == (num_ranges[col][0], num_ranges[col][1]):
                continue

            if min_v is not None:
                flt.reaction[f"{col}__ge"] = float(min_v)
            if max_v is not None:
                flt.reaction[f"{col}__le"] = float(max_v)

        # If no constraints and no datasets selected, use the full dataset
        if not flt.reaction and not datasets:
            filtered = full_ds
        else:
            filtered = flt.apply(full_ds)

    tags = set()
    for entry in filtered.reactions_df()["datasets_str"]:
        for t in str(entry).split(","):
            tags.add(t.strip())

    info = html.Span(
        [
            "Filtered dataset contains ",
            dbc.Badge(len(filtered), color="primary", className="me-1"),
            "reactions from ",
            dbc.Badge(len(tags), color="secondary", className="me-1"),
            "datasets, involving ",
            dbc.Badge(len(filtered.molecules_df()), color="success", className="me-1"),
            "unique molecules.",
        ]
    )
    stats = _model_stats_table(filtered)
    return filtered._rxn_ids, info, stats


def _build_figure(
    fig_type: str,
    idx_list,
    *,
    x=None,
    y=None,
    z=None,
    column=None,
    bins=None,
    model_name=None,
    color=None,
):
    """Construct the appropriate Plotly figure for the chosen type."""

    ds = Dataset(full_ds._arc, idx_list) if idx_list else full_ds
    if len(ds) == 0:
        return go.Figure()

    # guard against missing axes or NaN-only data
    if fig_type in {"TwoDRxn", "TwoDMol"} and (x is None or y is None):
        return go.Figure()
    if fig_type in {"ThreeDRxn", "ThreeDMol"} and (x is None or y is None or z is None):
        return go.Figure()

    if fig_type in {"TwoDRxn", "ThreeDRxn", "Histogram"}:
        df = ds.reactions_df()
    else:
        df = ds.molecules_df()
    cols = [c for c in [x, y, z, column] if c]
    missing = [c for c in cols if c not in df]
    if missing:
        return go.Figure()
    if cols and df[cols].dropna().empty:
        return go.Figure()
    model = _model(model_name)

    color_by = None
    if color and color != "None":
        if color == "Model Residual" and model:
            color_by = f"{model.name}_resid"
        else:
            color_by = color

    if fig_type == "TwoDRxn":
        fig = TwoDRxn(ds, x=x, y=y, model=model, color_by=color_by).figure
    elif fig_type == "TwoDMol":
        fig = TwoDMol(ds, x=x, y=y, color_by=color_by).figure
    elif fig_type == "ThreeDRxn":
        fig = ThreeDRxn(ds, x=x, y=y, z=z, model=model, color_by=color_by).figure
    elif fig_type == "ThreeDMol":
        fig = ThreeDMol(ds, x=x, y=y, z=z, color_by=color_by).figure
    else:
        fig = Histogram(
            ds, column=column, bins=bins, color_by=color_by, table="reactions"
        ).figure

    fig.update_layout(autosize=True, margin=dict(l=0, r=0, t=40, b=40))
    return fig


@app.callback(
    Output({"type": "fig", "pane": MATCH}, "figure"),
    Input({"type": "figtype", "pane": MATCH}, "value"),
    Input({"type": "ctrl", "pane": MATCH, "role": ALL}, "value"),
    Input("filtered-indexes", "data"),
)
def build_figure(fig_type, values, idx_list):
    items = dash.callback_context.inputs_list[1]
    args = {c["id"].get("role"): v for c, v in zip(items, values) if "role" in c["id"]}
    if not args:
        return go.Figure()
    return _build_figure(
        fig_type,
        idx_list,
        x=args.get("x"),
        y=args.get("y"),
        z=args.get("z"),
        column=args.get("col"),
        bins=args.get("bins"),
        model_name=args.get("model"),
        color=args.get("color"),
    )


@app.callback(
    Output({"type": "ctrl", "pane": MATCH, "role": "color"}, "options"),
    Input({"type": "ctrl", "pane": MATCH, "role": "model"}, "value"),
    State({"type": "figtype", "pane": MATCH}, "value"),
)
def toggle_residual_option(model_value, fig_type):
    if fig_type not in {"TwoDRxn", "ThreeDRxn"}:
        raise dash.exceptions.PreventUpdate

    disabled = model_value in (None, "None")
    base_opts = [
        {"label": "None", "value": "None"},
        {
            "label": "Model Residual",
            "value": "Model Residual",
            "disabled": disabled,
        },
    ]
    base_opts += [{"label": c, "value": c} for c in num_cols]
    return base_opts


@app.callback(
    Output({"type": "collapse", "pane": MATCH}, "is_open"),
    Input({"type": "opts-btn", "pane": MATCH}, "n_clicks"),
    State({"type": "collapse", "pane": MATCH}, "is_open"),
    prevent_initial_call=True,
)
def toggle_opts(n, is_open):
    return not is_open


@app.callback(
    Output("dataset-dropdown", "value"),
    *[Output(f"min-{cid}", "value") for cid in finite_ids],
    *[Output(f"max-{cid}", "value") for cid in finite_ids],
    Input("select-datasets-btn", "n_clicks"),
    Input("clear-filter-btn", "n_clicks"),
    prevent_initial_call=True,
)
def modify_dataset_values(select_n, clear_n):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "select-datasets-btn":
        return (
            all_tags,
            *[dash.no_update for _ in finite_ids],
            *[dash.no_update for _ in finite_ids],
        )

    # clear-filter-btn triggered
    return (
        None,
        *[num_ranges[c][0] for c in finite_cols],
        *[num_ranges[c][1] for c in finite_cols],
    )


if __name__ == "__main__":
    app.run(debug=True)

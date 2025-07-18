import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go

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


def dropdown(id_, options, value=None, multi=False):
    return dcc.Dropdown(
        options=[{"label": o, "value": o} for o in options],
        value=value,
        id=id_,
        multi=multi,
        clearable=False,
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
                    html.Button(
                        "Apply",
                        id="apply-filter-btn",
                        className="btn btn-primary btn-sm",
                    ),
                    html.Button(
                        "Clear",
                        id="clear-filter-btn",
                        className="btn btn-outline-secondary btn-sm ms-2",
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "position": "sticky",
                    "top": 0,
                    "zIndex": 1,
                    "background": "white",
                    "paddingBottom": "4px",
                },
            ),
            html.Label("Datasets"),
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
    )
    graph = dcc.Graph(
        id={"type": "fig", "pane": idx},
        style={"flex": 1, "width": "100%", "height": "100%"},
        config={"responsive": True},
    )
    controls = html.Div(id={"type": "controls", "pane": idx})
    return html.Div(
        [figure_type_dd, controls, graph],
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

    if fig_type in {"TwoDRxn", "TwoDMol"}:
        items = [
            html.Label("X"),
            DD("x", num_cols, "deltaG0"),
            html.Label("Y"),
            DD("y", num_cols, "computed_barrier"),
            html.Label("Color"),
            DD("color", ["None"] + list(num_cols), "None"),
        ]
        if "Rxn" in fig_type:
            items += [
                html.Label("Model"),
                DD("model", ["None"] + list(MODEL_OPTIONS), "None"),
            ]
        return items

    if fig_type in {"ThreeDRxn", "ThreeDMol"}:
        items = [
            html.Label("X"),
            DD("x", num_cols, "deltaG0"),
            html.Label("Y"),
            DD("y", num_cols, "asynchronicity"),
            html.Label("Z"),
            DD("z", num_cols, "computed_barrier"),
            html.Label("Color"),
            DD("color", ["None"] + list(num_cols), "None"),
        ]
        if "Rxn" in fig_type:
            items += [
                html.Label("Model"),
                DD("model", ["None"] + list(MODEL_OPTIONS), "None"),
            ]
        return items

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
    pane = dash.callback_context.triggered_id["pane"]
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
                "flex": "0 0 40%",
                "height": "100%",
                "minWidth": 0,
                "display": "grid",
                "gridTemplateRows": "3fr 2fr",
                "gap": "10px",
            },
        ),
        html.Div(
            [_figure_board()],
            style={
                "flex": "0 0 60%",
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
    [Output("filtered-indexes", "data"), Output("dataset-info", "children")],
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
    return filtered._rxn_ids, info


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
    State("filtered-indexes", "data"),
)
def build_figure(fig_type, values, idx_list):
    roles = [item["role"] for item in dash.callback_context.inputs_list[1]]
    args = dict(zip(roles, values))
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
    Output("dataset-dropdown", "value"),
    *[Output(f"min-{cid}", "value") for cid in finite_ids],
    *[Output(f"max-{cid}", "value") for cid in finite_ids],
    Input("clear-filter-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_all(n):
    return (
        None,
        *[num_ranges[c][0] for c in finite_cols],
        *[num_ranges[c][1] for c in finite_cols],
    )


if __name__ == "__main__":
    app.run(debug=True)

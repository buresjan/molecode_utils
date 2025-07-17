import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from molecode_utils.dataset import Dataset
from molecode_utils.filter import Filter
from molecode_utils.figures import TwoDRxn
from molecode_utils.model import ModelM1, ModelM2, ModelM3, ModelM4

# -----------------------------------------------------------------------------
# Dataset setup
# -----------------------------------------------------------------------------
DATA_PATH = "data/molecode-data-v0.1.0.h5"
full_ds = Dataset.from_hdf(DATA_PATH)

reaction_df = full_ds.reactions_df()
num_cols = reaction_df.select_dtypes(include="number").columns

safe_col_ids = {col: col.replace(".", "_") for col in num_cols}

all_tags = sorted(
    {
        tag.strip()
        for entry in reaction_df["datasets_str"]
        for tag in str(entry).split(",")
    }
)

# Pre-compute ranges for all numeric columns so we can build generic
# filtering sliders. Each slider spans the actual data range.
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
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Molecode Dashboard"


def _filter_tile():
    inputs = []
    for col in num_cols:
        cid = safe_col_ids[col]
        inputs.append(
            html.Div(
                [
                    html.Label(
                        [
                            f"{col} ",
                            html.I(
                                className="fa fa-circle-info text-secondary",
                                id=f"tip-{col}",
                                style={"cursor": "pointer"},
                            ),
                        ]
                    ),
                    dbc.Tooltip(
                        f"{num_ranges[col][0]} – {num_ranges[col][1]}",
                        target=f"tip-{col}",
                        placement="right",
                    ),
                    html.Div(
                        dcc.RangeSlider(
                            id=f"slider-{cid}",
                            min=num_ranges[col][0],
                            max=num_ranges[col][1],
                            value=list(num_ranges[col]),
                            allowCross=False,
                        ),
                        style={"marginTop": "4px"}  # ✅ Move style to wrapper Div
                    ),
                ]
            )
        )

    return html.Div(
        [
            html.H5("Dataset Filtering"),
            html.Button(
                "Apply Filter",
                id="apply-filter-btn",
                className="btn btn-primary btn-sm mb-2",
            ),
            html.Label("Datasets"),
            dropdown("dataset-dropdown", all_tags, multi=True),
            html.Div(inputs, style={"display": "grid", "rowGap": "8px"}),
        ],
        style={
            "border": "1px solid #ccc",
            "padding": "10px",
            "overflowY": "auto",
            "minHeight": "0",
        },
    )


def _graph_panel(idx: int) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Label("X"),
                    dropdown(f"x-select-{idx}", num_cols, value="deltaG0"),
                    html.Label("Y"),
                    dropdown(f"y-select-{idx}", num_cols, value="computed_barrier"),
                    html.Label("Color"),
                    dropdown(
                        f"color-select-{idx}",
                        ["None", "Model Residual"] + list(num_cols),
                        value="None",
                    ),
                    html.Label("Model"),
                    dropdown(
                        f"model-select-{idx}",
                        ["None"] + list(MODEL_OPTIONS),
                        value="None",
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "columnGap": "5px",
                },
            ),
            dcc.Graph(id=f"graph{idx}", style={"flex": 1, "width": "100%"}),
        ],
        style={
            "display": "flex",
            "flexDirection": "column",
            "flex": 1,
            "minHeight": "0",
        },
    )


def _graphs_tile():
    return html.Div(
        [_graph_panel(1), _graph_panel(2)],
        style={
            "display": "flex",
            "gap": "10px",
            "height": "100%",
            "border": "1px solid #ccc",
            "overflow": "hidden",
        },
    )


def _info_tile():
    return html.Div(
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
        style={"border": "1px solid #ccc", "padding": "10px"},
    )


def _analysis_tile():
    return html.Div(
        html.Div(
            "Model analysis coming soon...",
            style={"textAlign": "center", "color": "gray"},
        ),
        style={
            "border": "1px solid #ccc",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
        },
    )


app.layout = html.Div(
    [
        dcc.Store(id="filtered-indexes"),
        html.Div(
            [_filter_tile(), _info_tile()],
            style={
                "flex": 1,
                "height": "100%",
                "display": "grid",
                "gridTemplateRows": "3fr 2fr",
                "gap": "10px",
            },
        ),
        html.Div(
            [_graphs_tile(), _analysis_tile()],
            style={
                "flex": 1,
                "height": "100%",
                "display": "grid",
                "gridTemplateRows": "1fr 1fr",
                "gap": "10px",
            },
        ),
    ],
    style={"display": "flex", "height": "100vh", "gap": "10px", "padding": "10px"},
)


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------
filter_states = [State(f"slider-{safe_col_ids[c]}", "value") for c in num_cols]


@app.callback(
    [Output("filtered-indexes", "data"), Output("dataset-info", "children")],
    Input("apply-filter-btn", "n_clicks"),
    State("dataset-dropdown", "value"),
    *filter_states,
    prevent_initial_call=True,
)
def update_filters(n_clicks, datasets, *ranges):
    flt = Filter()
    if datasets:
        flt.datasets = datasets
    for col, rng in zip(num_cols, ranges):
        if rng is None:
            continue
        min_v, max_v = rng
        if min_v is not None:
            flt.reaction[f"{col}__ge"] = min_v
        if max_v is not None:
            flt.reaction[f"{col}__le"] = max_v

    filtered = flt.apply(full_ds)

    tags = set()
    for entry in filtered.reactions_df()["datasets_str"]:
        for t in str(entry).split(","):
            tags.add(t.strip())

    info = (
        f"Filtered dataset contains {len(filtered)} reactions from "
        f"{len(tags)} datasets, involving {len(filtered.molecules_df())} unique molecules."
    )
    return filtered._rxn_ids, info


def _build_figure(idx_list, x, y, model_name, color_var):
    ds = Dataset(full_ds._arc, idx_list) if idx_list else full_ds
    model = (
        MODEL_OPTIONS.get(model_name) if model_name and model_name != "None" else None
    )

    color_by = None
    if color_var and color_var != "None":
        if color_var == "Model Residual" and model:
            color_by = f"{model.name}_resid"
        elif color_var != "Model Residual":
            color_by = color_var

    fig = TwoDRxn(ds, x=x, y=y, model=model, color_by=color_by).figure
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=40), autosize=True)
    fig.update_layout(coloraxis_colorbar_x=1.02)
    return fig


@app.callback(
    Output("graph1", "figure"),
    [
        Input("filtered-indexes", "data"),
        Input("x-select-1", "value"),
        Input("y-select-1", "value"),
        Input("model-select-1", "value"),
        Input("color-select-1", "value"),
    ],
)
def refresh_graph1(idx_list, x, y, model_name, color_var):
    return _build_figure(idx_list, x, y, model_name, color_var)


@app.callback(
    Output("graph2", "figure"),
    [
        Input("filtered-indexes", "data"),
        Input("x-select-2", "value"),
        Input("y-select-2", "value"),
        Input("model-select-2", "value"),
        Input("color-select-2", "value"),
    ],
)
def refresh_graph2(idx_list, x, y, model_name, color_var):
    return _build_figure(idx_list, x, y, model_name, color_var)


if __name__ == "__main__":
    app.run(debug=True)

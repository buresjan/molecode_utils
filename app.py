import dash
from dash import dcc, html
from dash.dependencies import Input, Output

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
    "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Molecode Dashboard"

app.layout = html.Div(
    [
        dcc.Store(id="filtered-indexes"),
        # Upper left: filtering controls
        html.Div(
            [
                html.H5("Dataset Filtering"),
                html.Label("Datasets"),
                dropdown("dataset-dropdown", all_tags, multi=True),
                *[
                    html.Div(
                        [
                            html.Label(f"{col} range"),
                            dcc.RangeSlider(
                                num_ranges[col][0],
                                num_ranges[col][1],
                                step=(num_ranges[col][1] - num_ranges[col][0]) / 100
                                or 1,
                                value=list(num_ranges[col]),
                                marks=None,
                                id=f"slider-{col}",
                                tooltip={"placement": "bottom"},
                            ),
                            html.Br(),
                        ]
                    )
                    for col in num_cols
                ],
            ],
            style={"border": "1px solid #ccc", "padding": "10px", "overflowY": "auto"},
        ),
        # Upper right: two graphs
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("X"),
                                dropdown("x-select-1", num_cols, value="deltaG0"),
                                html.Label("Y"),
                                dropdown(
                                    "y-select-1", num_cols, value="computed_barrier"
                                ),
                                html.Label("Color"),
                                dropdown(
                                    "color-select-1",
                                    ["None", "Model Residual"] + list(num_cols),
                                    value="None",
                                ),
                                html.Label("Model"),
                                dropdown(
                                    "model-select-1",
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
                        dcc.Graph(id="graph1"),
                    ],
                    style={"width": "50%", "padding": "5px"},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("X"),
                                dropdown("x-select-2", num_cols, value="deltaG0"),
                                html.Label("Y"),
                                dropdown(
                                    "y-select-2", num_cols, value="computed_barrier"
                                ),
                                html.Label("Color"),
                                dropdown(
                                    "color-select-2",
                                    ["None", "Model Residual"] + list(num_cols),
                                    value="None",
                                ),
                                html.Label("Model"),
                                dropdown(
                                    "model-select-2",
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
                        dcc.Graph(id="graph2"),
                    ],
                    style={"width": "50%", "padding": "5px"},
                ),
            ],
            style={
                "display": "flex",
                "height": "100%",
                "border": "1px solid #ccc",
                "overflow": "hidden",
            },
        ),
        # Lower left: dataset info and lookups
        html.Div(
            [
                dcc.Tabs(
                    [
                        dcc.Tab(html.Div(id="dataset-info"), label="Dataset Info"),
                        dcc.Tab(
                            html.Div("Molecule lookup coming soon"),
                            label="Molecule Lookup",
                        ),
                        dcc.Tab(
                            html.Div("Reaction lookup coming soon"),
                            label="Reaction Lookup",
                        ),
                    ]
                )
            ],
            style={"border": "1px solid #ccc", "padding": "10px", "overflowY": "auto"},
        ),
        # Lower right: placeholder for model analysis
        html.Div(
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
        ),
    ],
    style={
        "display": "grid",
        "gridTemplateColumns": "1fr 1fr",
        "gridTemplateRows": "1fr 1fr",
        "gap": "10px",
        "height": "100vh",
        "padding": "10px",
    },
)


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------
filter_inputs = [Input(f"slider-{c}", "value") for c in num_cols]


@app.callback(
    [Output("dataset-info", "children"), Output("filtered-indexes", "data")],
    [Input("dataset-dropdown", "value")] + filter_inputs,
)
def update_filters(datasets, *ranges):
    flt = Filter()
    if datasets:
        flt.datasets = datasets
    for col, rng in zip(num_cols, ranges):
        if rng:
            flt.reaction[f"{col}__ge"] = rng[0]
            flt.reaction[f"{col}__le"] = rng[1]

    filtered = flt.apply(full_ds)

    tags = set()
    for entry in filtered.reactions_df()["datasets_str"]:
        for t in str(entry).split(","):
            tags.add(t.strip())

    info = (
        f"Filtered dataset contains {len(filtered)} reactions from "
        f"{len(tags)} datasets, involving {len(filtered.molecules_df())} unique molecules."
    )
    return info, filtered._rxn_ids


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

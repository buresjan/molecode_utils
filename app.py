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

DG0_MIN = reaction_df["deltaG0"].min()
DG0_MAX = reaction_df["deltaG0"].max()
BAR_MIN = reaction_df["computed_barrier"].min()
BAR_MAX = reaction_df["computed_barrier"].max()
SEB_MIN = reaction_df["oxidant.self_exchange_barrier"].min()
SEB_MAX = reaction_df["oxidant.self_exchange_barrier"].max()


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
                html.Br(),
                html.Label("deltaG0 range"),
                dcc.RangeSlider(
                    DG0_MIN,
                    DG0_MAX,
                    step=1,
                    value=[DG0_MIN, DG0_MAX],
                    marks=None,
                    id="dg0-slider",
                    tooltip={"placement": "bottom"},
                ),
                html.Br(),
                html.Label("computed_barrier range"),
                dcc.RangeSlider(
                    BAR_MIN,
                    BAR_MAX,
                    step=1,
                    value=[BAR_MIN, BAR_MAX],
                    marks=None,
                    id="barrier-slider",
                    tooltip={"placement": "bottom"},
                ),
                html.Br(),
                html.Label("oxidant.self_exchange_barrier range"),
                dcc.RangeSlider(
                    SEB_MIN,
                    SEB_MAX,
                    step=1,
                    value=[SEB_MIN, SEB_MAX],
                    marks=None,
                    id="seb-slider",
                    tooltip={"placement": "bottom"},
                ),
            ],
            style={"border": "1px solid #ccc", "padding": "10px", "overflowY": "auto"},
        ),
        # Upper right: two graphs
        html.Div(
            [
                html.Div(
                    [
                        html.Label("X"),
                        dropdown("x-select-1", num_cols, value="deltaG0"),
                        html.Label("Y"),
                        dropdown("y-select-1", num_cols, value="computed_barrier"),
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
                        dcc.Graph(id="graph1"),
                    ],
                    style={"width": "50%", "padding": "5px"},
                ),
                html.Div(
                    [
                        html.Label("X"),
                        dropdown("x-select-2", num_cols, value="deltaG0"),
                        html.Label("Y"),
                        dropdown("y-select-2", num_cols, value="computed_barrier"),
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
@app.callback(
    [Output("dataset-info", "children"), Output("filtered-indexes", "data")],
    [
        Input("dataset-dropdown", "value"),
        Input("dg0-slider", "value"),
        Input("barrier-slider", "value"),
        Input("seb-slider", "value"),
    ],
)
def update_filters(datasets, dg0_range, barrier_range, seb_range):
    flt = Filter()
    if datasets:
        flt.datasets = datasets
    if dg0_range:
        flt.reaction["deltaG0__ge"] = dg0_range[0]
        flt.reaction["deltaG0__le"] = dg0_range[1]
    if barrier_range:
        flt.reaction["computed_barrier__ge"] = barrier_range[0]
        flt.reaction["computed_barrier__le"] = barrier_range[1]
    if seb_range:
        flt.oxidant["self_exchange_barrier__ge"] = seb_range[0]
        flt.oxidant["self_exchange_barrier__le"] = seb_range[1]

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
        else:
            color_by = color_var
    elif model and color_var == "Model Residual":
        color_by = f"{model.name}_resid"

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
    app.run_server(debug=True)

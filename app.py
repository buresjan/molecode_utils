import dash
from dash import dcc, html

from molecode_utils.dataset import Dataset
from molecode_utils.filter import Filter
from molecode_utils.figures import TwoDRxn
from molecode_utils.model import ModelM1, ModelM2, ModelM3, ModelM4

# load dataset
DATA_PATH = "data/molecode-data-v0.1.0.h5"
full_ds = Dataset.from_hdf(DATA_PATH)

# compute helper lists
reaction_df = full_ds.reactions_df()
num_cols = reaction_df.select_dtypes(include="number").columns
# dataset tags (split comma-separated lists)
all_tags = sorted(
    {
        tag.strip()
        for entry in reaction_df["datasets_str"]
        for tag in str(entry).split(",")
    }
)

external_stylesheets = [
    "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Molecode Dashboard"
app.config.suppress_callback_exceptions = True

# slider ranges
DG0_MIN = reaction_df["deltaG0"].min()
DG0_MAX = reaction_df["deltaG0"].max()
BAR_MIN = reaction_df["computed_barrier"].min()
BAR_MAX = reaction_df["computed_barrier"].max()
SEB_MIN = reaction_df["oxidant.self_exchange_barrier"].min()
SEB_MAX = reaction_df["oxidant.self_exchange_barrier"].max()


def make_dropdown(id_, options, value=None, multi=False):
    return dcc.Dropdown(
        options=[{"label": o, "value": o} for o in options],
        value=value,
        id=id_,
        multi=multi,
        clearable=False,
    )


# -------------------------------------------------------------------
# Layout
# -------------------------------------------------------------------
app.layout = html.Div(
    [
        dcc.Store(id="filtered-indexes"),
        html.Div(
            [
                html.Div(
                    [
                        html.H4("Dataset Filtering"),
                        html.Label("Datasets"),
                        make_dropdown("dataset-dropdown", all_tags, multi=True),
                        html.Br(),
                        html.Label("deltaG0 range"),
                        dcc.RangeSlider(
                            DG0_MIN,
                            DG0_MAX,
                            step=1,
                            value=[DG0_MIN, DG0_MAX],
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
                            id="seb-slider",
                            tooltip={"placement": "bottom"},
                        ),
                    ],
                    style={"width": "50%", "padding": "10px"},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("X variable"),
                                        make_dropdown(
                                            "x-select-1", num_cols, value="deltaG0"
                                        ),
                                        html.Label("Y variable"),
                                        make_dropdown(
                                            "y-select-1",
                                            num_cols,
                                            value="computed_barrier",
                                        ),
                                        html.Label("Model"),
                                        make_dropdown(
                                            "model-select-1",
                                            ["None", "M1", "M2", "M3", "M4"],
                                            value="None",
                                        ),
                                    ]
                                ),
                                dcc.Graph(id="graph1"),
                            ],
                            style={"width": "100%"},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("X variable"),
                                        make_dropdown(
                                            "x-select-2", num_cols, value="deltaG0"
                                        ),
                                        html.Label("Y variable"),
                                        make_dropdown(
                                            "y-select-2",
                                            num_cols,
                                            value="computed_barrier",
                                        ),
                                        html.Label("Model"),
                                        make_dropdown(
                                            "model-select-2",
                                            ["None", "M1", "M2", "M3", "M4"],
                                            value="None",
                                        ),
                                    ]
                                ),
                                dcc.Graph(id="graph2"),
                            ],
                            style={"width": "100%"},
                        ),
                    ],
                    style={"width": "50%", "padding": "10px"},
                ),
            ],
            style={"display": "flex"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Tabs(
                            [
                                dcc.Tab(
                                    html.Div(id="dataset-info"), label="Dataset Info"
                                ),
                                dcc.Tab(html.Div("..."), label="Molecule Lookup"),
                                dcc.Tab(html.Div("..."), label="Reaction Lookup"),
                            ]
                        )
                    ],
                    style={"width": "50%", "padding": "10px"},
                ),
                html.Div(html.Div(id="placeholder"), style={"width": "50%"}),
            ],
            style={"display": "flex"},
        ),
    ],
    style={"padding": "20px"},
)


# -------------------------------------------------------------------
# Callbacks
# -------------------------------------------------------------------
@app.callback(
    [dash.Output("dataset-info", "children"), dash.Output("filtered-indexes", "data")],
    [
        dash.Input("dataset-dropdown", "value"),
        dash.Input("dg0-slider", "value"),
        dash.Input("barrier-slider", "value"),
        dash.Input("seb-slider", "value"),
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
    info = f"{len(filtered)} reactions ({len(filtered.molecules_df())} molecules) selected."
    return info, filtered._rxn_ids


def _build_figure(idx_list, x, y, model_name):
    ds = Dataset(full_ds._arc, idx_list) if idx_list else full_ds
    model = None
    if model_name and model_name != "None":
        model_map = {
            "M1": ModelM1(),
            "M2": ModelM2(),
            "M3": ModelM3(),
            "M4": ModelM4(),
        }
        model = model_map.get(model_name)
    color_by = f"{model.name}_resid" if model else None
    fig = TwoDRxn(ds, x=x, y=y, model=model, color_by=color_by).figure
    return fig


@app.callback(
    dash.Output("graph1", "figure"),
    [
        dash.Input("filtered-indexes", "data"),
        dash.Input("x-select-1", "value"),
        dash.Input("y-select-1", "value"),
        dash.Input("model-select-1", "value"),
    ],
)
def refresh_graph1(idx_list, x, y, model_name):
    return _build_figure(idx_list, x, y, model_name)


@app.callback(
    dash.Output("graph2", "figure"),
    [
        dash.Input("filtered-indexes", "data"),
        dash.Input("x-select-2", "value"),
        dash.Input("y-select-2", "value"),
        dash.Input("model-select-2", "value"),
    ],
)
def refresh_graph2(idx_list, x, y, model_name):
    return _build_figure(idx_list, x, y, model_name)


if __name__ == "__main__":
    app.run_server(debug=True)

"""Minimal Dash demo using molecode_utils with a safe component ID."""

from __future__ import annotations

import pathlib

import dash
from dash import Dash, dcc, html

from molecode_utils import Dataset, sanitize_id

H5_PATH = pathlib.Path("data/molecode-data-v0.1.0.h5")

app = Dash(__name__)

slider_id = sanitize_id("slider-oxidant.0O")

app.layout = html.Div(
    [
        dcc.Slider(id=slider_id, min=0, max=10, step=1, value=5),
        html.Div(id="output")
    ]
)

@app.callback(dash.dependencies.Output("output", "children"),
              [dash.dependencies.Input(slider_id, "value")])
def display_value(val):
    return f"Selected value: {val}"

if __name__ == "__main__":
    # The dataset is not used directly; loading is shown for completeness
    with Dataset.from_hdf(H5_PATH) as ds:
        print(f"Dataset contains {len(ds)} reactions")
    app.run_server(debug=True)

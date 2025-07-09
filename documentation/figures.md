# Figure Helpers Cheatsheet

`molecode_utils.figures` provides simple helpers for quick visualisations using
either Plotly or Matplotlib.

## TwoDRxn

`TwoDRxn` creates a scatter plot of reaction-level variables.

```python
from molecode_utils.dataset import Dataset
from molecode_utils.model import ModelM4
from molecode_utils.figures import TwoDRxn

ds = Dataset.from_hdf('data/molecode-data-v0.1.0.h5')
fig = TwoDRxn(
    ds,
    x='deltaG0',
    y='computed_barrier',
    model=ModelM4(),
    color_by='dataset_main',
)
fig.figure.write_html('scatter.html')
fig.show()
ds.close()
```

Arguments:

- `dataset` – `Dataset` instance
- `x`, `y` – column names from `reactions_df()`
- `model` – optional `Model` to include predictions/residuals
- `color_by` / `group_by` – column used for colouring / faceting
- `latex_labels` – switch between LaTeX and plain text labels
- `backend` – `'plotly'` (default) or `'matplotlib'` for the rendering engine

Depending on the chosen backend the helper exposes either a
`plotly.graph_objects.Figure` or a `matplotlib.figure.Figure` as `.figure` for
further customisation.

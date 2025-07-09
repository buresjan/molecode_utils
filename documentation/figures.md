# Figure Helpers Cheatsheet

`molecode_utils.figures` provides simple Plotly based helpers for quick visualisations.

## TwoDRxn

`TwoDRxn` creates a scatter plot of reaction-level variables.

```python
from molecode_utils.dataset import Dataset
from molecode_utils.model import ModelM4
from molecode_utils.figures import TwoDRxn

ds = Dataset.from_hdf('data/molecode-data-v0.1.0.h5')
fig = TwoDRxn(ds, x='deltaG0', y='computed_barrier', model=ModelM4(), color_by='dataset_main')
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

The helper exposes the underlying `plotly.graph_objects.Figure` as `.figure` for further customisation.

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

Note that Plotly does not render LaTeX in 3D figures. The `ThreeDRxn` and
`ThreeDMol` helpers therefore always use plain text labels that include the unit
in square brackets.

When colouring points with the Matplotlib backend the colour bar is labelled with the variable name.

Depending on the chosen backend the helper exposes either a
`plotly.graph_objects.Figure` or a `matplotlib.figure.Figure` as `.figure` for
further customisation.

## TwoDMol

`TwoDMol` is the molecule analogue of `TwoDRxn`. It plots any two
molecule-level columns returned by `Dataset.molecules_df()` in a 2D scatter
plot.

```python
from molecode_utils.dataset import Dataset
from molecode_utils.figures import TwoDMol

ds = Dataset.from_hdf('data/molecode-data-v0.1.0.h5')
fig = TwoDMol(
    ds,
    x='pKaRH',
    y='E_H',
    color_by='omega',
)
fig.figure.write_html('mol_scatter.html')
fig.show()
ds.close()
```

Arguments are the same as for `TwoDRxn` except there is no `model` option.
You can colour points by any numeric column or facet the plot by a categorical
field (e.g. the primary dataset tag via `group_by='dataset_main'`).  Both
Plotly and Matplotlib backends are supported.

## ThreeDRxn

`ThreeDRxn` extends `TwoDRxn` into the third dimension and plots three
reaction-level variables in a 3D scatter plot.

```python
from molecode_utils.dataset import Dataset
from molecode_utils.model import ModelM4
from molecode_utils.figures import ThreeDRxn

ds = Dataset.from_hdf('data/molecode-data-v0.1.0.h5')
fig = ThreeDRxn(
    ds,
    x='asynchronicity',
    y='frustration',
    z='deltaG0',
    model=ModelM4(),
    color_by='dataset_main',
)
fig.figure.write_html('rxn_scatter3d.html')
fig.show()
ds.close()
```

Arguments are the same as for `TwoDRxn` with the additional `z` variable.
Both Plotly and Matplotlib backends are available. Axes labels are formatted in
plain text with units and the default figure size is larger to fit the longer
titles.

## ThreeDMol

`ThreeDMol` mirrors `TwoDMol` for molecules but in 3D.

```python
from molecode_utils.dataset import Dataset
from molecode_utils.figures import ThreeDMol

ds = Dataset.from_hdf('data/molecode-data-v0.1.0.h5')
fig = ThreeDMol(
    ds,
    x='pKaRH',
    y='E_H',
    z='omega',
    group_by='dataset_main',
)
fig.figure.write_html('mol_scatter3d.html')
fig.show()
ds.close()
```

Arguments mirror `TwoDMol` with an extra `z` variable.
Like `ThreeDRxn`, the labels are always rendered as plain text with units and the
figure size is increased when using Plotly.

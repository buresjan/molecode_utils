import pathlib
import plotly.io as pio

from molecode_utils.dataset import Dataset
from molecode_utils.model import ModelM4
from molecode_utils.figures import TwoDRxn

# Paths
H5_PATH = pathlib.Path("data/molecode-data-v0.1.0.h5")
ASSETS = pathlib.Path("examples/assets")
ASSETS.mkdir(exist_ok=True)

# Load dataset
ds = Dataset.from_hdf(H5_PATH)

pio.renderers.default = "browser"

# Basic scatter with LaTeX labels
fig = TwoDRxn(ds, x="deltaG0", y="computed_barrier", latex_labels=True)
fig.figure.write_html(ASSETS / "basic_scatter.html")
fig.show()

# Colored by asynchronicity
fig = TwoDRxn(
    ds,
    x="deltaG0",
    y="computed_barrier",
    color_by="asynchronicity",
    latex_labels=True,
)
fig.figure.write_html(ASSETS / "color_by_async.html")
fig.show()

# Including model predictions and plain-text labels
m = ModelM4()
fig = TwoDRxn(
    ds,
    x="computed_barrier",
    y=f"{m.name}_pred",
    model=m,
    group_by="datasets_str",
    latex_labels=False,
)
fig.figure.write_html(ASSETS / "with_model.html")
fig.show()

# Matplotlib backend
fig = TwoDRxn(
    ds,
    x="deltaG0",
    y="computed_barrier",
    backend="matplotlib",
)
fig.figure.savefig(ASSETS / "basic_scatter.png")
fig.show()
fig.show()

# Matplotlib backend
fig = TwoDRxn(
    ds,
    x="deltaG0",
    y="computed_barrier",
    color_by="asynchronicity",
    backend="matplotlib",
)
fig.figure.savefig(ASSETS / "colored_scatter.png")
fig.show()

# Cleanupds.close()

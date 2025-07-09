import pathlib
import plotly.io as pio

from molecode_utils.dataset import Dataset
from molecode_utils.figures import Histogram

H5_PATH = pathlib.Path("data/molecode-data-v0.1.0.h5")
ASSETS = pathlib.Path("examples/assets")
ASSETS.mkdir(exist_ok=True)

pio.renderers.default = "browser"

ds = Dataset.from_hdf(H5_PATH)

# Reaction level histogram using Plotly
fig = Histogram(ds, column="deltaG0", bins=40, color_by="dataset_main")
fig.figure.write_html(ASSETS / "hist_rxn_plotly.html")
fig.show()

# Molecule level histogram using Matplotlib
fig = Histogram(
    ds,
    column="E_H",
    table="molecules",
    bins=40,
    color_by="dataset_main",
    backend="matplotlib",
)
fig.figure.savefig(ASSETS / "hist_mol_matplotlib.png")
fig.show()

ds.close()

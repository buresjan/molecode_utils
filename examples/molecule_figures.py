import pathlib
import plotly.io as pio

from molecode_utils.dataset import Dataset
from molecode_utils.figures import TwoDMol

H5_PATH = pathlib.Path("data/molecode-data-v0.1.0.h5")
ASSETS = pathlib.Path("examples/assets")
ASSETS.mkdir(exist_ok=True)

pio.renderers.default = "browser"

ds = Dataset.from_hdf(H5_PATH)

# Plot two basic variables
fig = TwoDMol(ds, x="pKaRH", y="E_H")
fig.figure.write_html(ASSETS / "mol_basic.html")
fig.show()

# Colour by electrophilicity and facet by dataset
fig = TwoDMol(
    ds,
    x="pKaRH",
    y="E_H",
    color_by="omega",
    group_by="dataset_main",
)
fig.figure.write_html(ASSETS / "mol_color_group.html")
fig.show()

# Matplotlib backend
fig = TwoDMol(ds, x="pKaRH", y="E_H", backend="matplotlib")
fig.figure.savefig(ASSETS / "mol_matplotlib.png")
fig.show()

ds.close()

import pathlib
import plotly.io as pio

from molecode_utils.dataset import Dataset
from molecode_utils.model import ModelM4
from molecode_utils.figures import ThreeDRxn, ThreeDMol

H5_PATH = pathlib.Path("data/molecode-data-v0.1.0.h5")
ASSETS = pathlib.Path("examples/assets")
ASSETS.mkdir(exist_ok=True)

pio.renderers.default = "browser"

ds = Dataset.from_hdf(H5_PATH)

# Reaction level 3D scatter
fig = ThreeDRxn(
    ds,
    x="asynchronicity",
    y="frustration",
    z="deltaG0",
    model=ModelM4(),
    color_by="dataset_main",
)
fig.figure.write_html(ASSETS / "rxn_scatter3d.html")
fig.show()

# Molecule level 3D scatter
fig = ThreeDMol(
    ds,
    x="pKaRH",
    y="E_H",
    z="omega",
    group_by="dataset_main",
)
fig.figure.write_html(ASSETS / "mol_scatter3d.html")
fig.show()

ds.close()

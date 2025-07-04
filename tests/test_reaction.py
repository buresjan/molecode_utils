import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pathlib
import h5py
import pytest
from molecode_utils.molecule import Molecule
from molecode_utils.reaction import Reaction

DATA_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "molecode-data-v0.1.0.h5"


def _load_lookup(h5):
    mds = h5["molecules"]
    units = mds.attrs["column_units"]
    return {
        int(row["mol_idx"]): Molecule.from_row(row, units)
        for row in mds[:5]
    }


def test_reaction_from_row_basic():
    with h5py.File(DATA_PATH, "r") as h5:
        lookup = _load_lookup(h5)
        rds = h5["reactions"]
        units = rds.attrs["column_units"]
        row = rds[0]
        rxn = Reaction.from_row(row, units, molecule_lookup=lookup)

    assert rxn.id == int(row["rxn_idx"])
    assert rxn.oxidant.id == int(row["oxid_idx"])
    assert rxn.substrate.id == int(row["subst_idx"])
    # dynamic attribute access
    assert pytest.approx(float(rxn.deltaG0)) == float(row["deltaG0"])

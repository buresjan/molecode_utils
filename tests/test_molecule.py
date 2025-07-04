import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pytest
import pathlib
import h5py
from molecode_utils.molecule import Molecule, UnitList

DATA_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "molecode-data-v0.1.0.h5"


def test_molecule_from_row_basic():
    with h5py.File(DATA_PATH, "r") as h5:
        ds = h5["molecules"]
        units = ds.attrs["column_units"]
        row = ds[0]
        mol = Molecule.from_row(row, units)

    assert mol.id == int(row["mol_idx"])
    assert mol.smiles.value == (row["smiles"].decode().rstrip() if isinstance(row["smiles"], (bytes, bytearray)) else row["smiles"])
    # dataset column should become UnitList
    assert isinstance(mol.dataset, UnitList)

    d = mol.to_dict()
    s = mol.as_series()
    for k in d:
        assert d[k] == s[k]

    # immutability
    with pytest.raises(AttributeError):
        mol.some_new_attr = 1

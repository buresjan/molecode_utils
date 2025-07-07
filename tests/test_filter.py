import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pathlib

from molecode_utils.dataset import Dataset
from molecode_utils.filter import Filter

DATA_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "molecode-data-v0.1.0.h5"


def test_blank_filter_keeps_dataset():
    ds = Dataset.from_hdf(DATA_PATH)
    try:
        flt = Filter()
        out = flt(ds)
        assert len(out) == len(ds)
        assert set(out._rxn_ids) == set(ds._rxn_ids)
    finally:
        ds.close()


def test_nonsense_filter_returns_empty():
    ds = Dataset.from_hdf(DATA_PATH)
    try:
        flt = Filter(reaction={"computed_barrier__lt": -10e6})
        out = flt(ds)
        assert len(out) == 0
    finally:
        ds.close()


def test_symbolic_operator_parsing():
    ds = Dataset.from_hdf(DATA_PATH)
    try:
        flt = Filter(oxidant={"E_H >": 1.0})
        out1 = flt(ds)
        out2 = ds.filter(**{"oxidant.E_H__gt": 1.0})
        assert set(out1._rxn_ids) == set(out2._rxn_ids)
    finally:
        ds.close()


def test_dataset_tag_filter():
    ds = Dataset.from_hdf(DATA_PATH)
    try:
        flt = Filter(datasets=["Phenols"])
        out1 = flt(ds)
        out2 = ds.filter(datasets=["Phenols"])
        assert set(out1._rxn_ids) == set(out2._rxn_ids)
    finally:
        ds.close()


def test_equality_filter_on_molecule_id():
    ds = Dataset.from_hdf(DATA_PATH)
    try:
        first_id = ds.reactions_df().loc[0, "oxidant.molecule_id"]
        flt = Filter(oxidant={"molecule_id =": first_id})
        out = flt(ds)
        assert len(out) > 0
        ids = {
            ds.get_reaction(i).oxidant.molecule_id.value for i in out._rxn_ids
        }
        assert ids == {first_id.decode()}
    finally:
        ds.close()

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pathlib
import pytest
from molecode_utils.dataset import Dataset

DATA_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "molecode-data-v0.1.0.h5"


def test_dataset_basic_operations():
    ds = Dataset.from_hdf(DATA_PATH)
    try:
        # length matches reactions_df rows
        rdf = ds.reactions_df()
        assert len(ds) == len(rdf)
        assert "datasets_str" in rdf.columns

        mdf = ds.molecules_df()
        assert len(mdf) > 0
        assert "mol_idx" in mdf.columns

        rxn = ds[0]
        assert rxn.id == rdf.iloc[0]["rxn_idx"]

        # simple filtering
        filt = ds.filter(deltaG0__lt=0)
        assert 0 < len(filt) <= len(ds)
        for r in filt:
            assert float(r.deltaG0) < 0

        # filt_vec = ds.filter(**{"oxidant.E_H__gt": 1.0})
        # filt_lambda = ds.filter(func=lambda r: float(r.oxidant.E_H) > 1.0)
        # assert len(filt_vec) == len(filt_lambda)
        # assert set(filt_vec._rxn_ids) == set(filt_lambda._rxn_ids)
    finally:
        ds.close()

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pathlib
from molecode_utils.dataset import Dataset
from molecode_utils.model import ModelS, ModelM1, ModelM2, ModelM3, ModelM4

DATA_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "molecode-data-v0.1.0.h5"


def test_model_predictions_residuals():
    ds = Dataset.from_hdf(DATA_PATH)
    try:
        rxn = ds[0]
        models = [ModelS(lambda_000=20), ModelM1(), ModelM2(), ModelM3(), ModelM4()]
        for m in models:
            pred = m.predict(rxn)
            resid = m.residual(rxn)
            assert abs(resid - (pred - float(rxn.computed_barrier))) < 1e-8

        subset = ds[:5]
        for m in models:
            df = m.evaluate(subset)
            assert "residual" in df.columns
            assert "MAE" in df.attrs and "RMSE" in df.attrs
    finally:
        ds.close()

import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from molecode_utils.figures import TwoDRxn
from molecode_utils.var_metadata import variable_metadata


def test_make_label_unicode():
    label = TwoDRxn._make_label("computed_barrier", latex=False)
    assert label == "ΔG‡ [kcal/mol]"

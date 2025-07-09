import pathlib
import time
from molecode_utils.dataset import Dataset
from molecode_utils.model import ModelM4
from molecode_utils.figures import TwoDRxn

H5_PATH = pathlib.Path('data/molecode-data-v0.1.0.h5')

# load data
ds = Dataset.from_hdf(H5_PATH)
model = ModelM4()

start = time.perf_counter()
TwoDRxn(ds, x='computed_barrier', y=f'{model.name}_pred', model=model, fast_predict=False)
slow = time.perf_counter() - start

start = time.perf_counter()
TwoDRxn(ds, x='computed_barrier', y=f'{model.name}_pred', model=model)
fast = time.perf_counter() - start

print(f"slow mode: {slow:.2f}s")  # up to 150s
print(f"fast mode: {fast:.2f}s")

ds.close()
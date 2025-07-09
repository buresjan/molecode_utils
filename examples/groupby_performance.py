import pathlib
import time
from molecode_utils.dataset import Dataset
from molecode_utils.model import ModelM4
from molecode_utils.figures import TwoDRxn

H5_PATH = pathlib.Path('data/molecode-data-v0.1.0.h5')

# Load dataset
ds = Dataset.from_hdf(H5_PATH)
model = ModelM4()

start = time.perf_counter()
TwoDRxn(ds, x='computed_barrier', y=f'{model.name}_pred', model=model, group_by='datasets_str')
slow = time.perf_counter() - start

start = time.perf_counter()
TwoDRxn(ds, x='computed_barrier', y=f'{model.name}_pred', model=model, group_by='dataset_main')
fast = time.perf_counter() - start

print(f"group_by datasets_str: {slow:.2f}s")
print(f"group_by dataset_main: {fast:.2f}s")

ds.close()
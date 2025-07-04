# Running the test-suite

The project uses **pytest**. Ensure the package dependencies are installed
(e.g. `pip install -e .`) and run the tests from the repository root:

```bash
pytest -v
```

The tests expect the bundled HDF5 example archive at `data/molecode-data-v0.1.0.h5`
and load it directly.

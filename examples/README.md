# Tutorial scripts

This folder contains short, self-contained Python scripts that demonstrate how
to work with the `molecode_utils` package. Each file focuses on a specific part
of the API and can be run directly from the project root, e.g.
`python examples/dataset_tutorial.py`.

- **`dataset_tutorial.py`** – loads the example archive and showcases Dataset filtering, slicing and exporting features.
- **`molecule_tutorial.py`** – inspects a single molecule entry and illustrates unit-aware arithmetic.
- **`reaction_tutorial.py`** – explores reaction data and common access patterns.
- **`model_tutorial.py`** – runs a few of the included Marcus-type models on sample reactions.
- **`sequential_analysis.py`** – end-to-end example that chains
  :meth:`Dataset.filter` calls for sequential data reduction,
  plotting and `ModelM4` evaluation.
- **`filter_tutorial.py`** – step-by-step guide to the :class:`Filter` helper.
- **`molecule_figures.py`** – demonstrates the :class:`TwoDMol` scatter helper.
- **`three_d_figures.py`** – showcases the :class:`ThreeDRxn` and :class:`ThreeDMol` helpers.
- **`histogram_tutorial.py`** – demonstrates the :class:`Histogram` helper on both backends.

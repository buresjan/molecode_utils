# Filter Cheatsheet

`molecode_utils.filter.Filter` is a dataclass that encapsulates arguments for `Dataset.filter`.
It allows you to build reusable filtering logic and apply it to multiple datasets.

## Creating a Filter

```python
from molecode_utils.filter import Filter

# keep reactions with barrier > 10 kcal/mol coming from the 'Phenols' dataset
f = Filter(
    query=None,
    datasets=['Phenols'],
    reaction={'computed_barrier >': 10},
)
```

Keyword groups correspond to reaction, oxidant, and substrate columns. Operators can be written with symbolic forms (`>` `<=` …) or with the explicit `__gt`, `__le` suffixes.

```python
# oxidant E_H >= 1.0 and substrate pKaRH <= 15
f2 = Filter(oxidant={'E_H >=': 1.0}, substrate={'pKaRH <=': 15})
```

## Applying filters

```python
subset = f(dataset)        # identical to f.apply(dataset)
```

`Filter.apply` simply forwards all criteria to `Dataset.filter`.
Filters can be composed by passing the output dataset of one filter to another.

# Filter Cheatsheet

`molecode_utils.filter.Filter` is a small dataclass that mirrors the arguments of `Dataset.filter`. It keeps track of every criterion so that the same filtering logic can be applied to many datasets.

A `Filter` can specify:

- a pandas style `query` evaluated on the reactions dataframe
- a list of dataset tag substrings via `datasets`
- column inequalities for `reaction`, `oxidant` and `substrate` fields
- an optional `func` lambda that receives each `Reaction`

Calling the filter returns a new `Dataset` with only the reactions that satisfy **all** criteria.

## Creating a Filter

```python
from molecode_utils.dataset import Dataset
from molecode_utils.filter import Filter

with Dataset.from_hdf('data/molecode-data-v0.1.0.h5') as ds:
    # reactions coming from the Phenols dataset with computed barrier > 10 kcal/mol
    f = Filter(datasets=['Phenols'], reaction={'computed_barrier >': 10})
    subset = f(ds)
    print(len(subset))
```

Keyword groups correspond to the reaction table and the two molecule tables. Operators may use symbolic forms (`>`, `<=`, ... ) or the explicit suffixes (`__gt`, `__le`, ...).

```python
# oxidant E_H >= 1.0 and substrate pKaRH <= 15
f2 = Filter(
    oxidant={'E_H >=': 1.0},
    substrate={'pKaRH <=': 15},
)
```

## Query expressions

The `query` string is evaluated with `pandas.DataFrame.eval` on the reaction dataframe. This is convenient for compound logical expressions.

```python
f3 = Filter(query="(computed_barrier > 10) & (deltaG0 < 0)")
matched = f3(ds)
```

String columns must be backtick quoted if they contain spaces or punctuation.

```python
f4 = Filter(query="`oxidant.name` == 'TEMPO'")
```

## Dataset tags

Filtering by dataset tags checks the `datasets_str` column for substring matches. Multiple tags are ORed together.

```python
# keep reactions belonging to any dataset containing "Phenols" or "Amines"
phenols_or_amines = Filter(datasets=['Phenols', 'Amines'])
subset = phenols_or_amines(ds)
```

## Column inequalities

`reaction`, `oxidant` and `substrate` accept dictionaries where keys are column names with a comparison operator. Without an operator the value is compared for equality.

```python
# computed_barrier >= 15
base = Filter(reaction={'computed_barrier__ge': 15})

# same using symbolic operator
base2 = Filter(reaction={'computed_barrier >=': 15})
```

The available suffixes are `__lt`, `__le`, `__gt`, `__ge`, `__eq` and `__ne`.

## Lambda functions

For custom logic that cannot be expressed with column comparisons, supply a `func` callable. It receives each `Reaction` instance in turn.

```python
def only_full_data(r: Reaction) -> bool:
    return (r.computed_barrier is not None) and (r.deltaG0 is not None)

custom = Filter(func=only_full_data)
```

Inside the lambda you may access all `Reaction` fields and the nested `Molecule` objects.

## Combining filters

Filters can be composed by applying them sequentially. Each call returns a new dataset view which can be passed to another filter.

```python
f_a = Filter(reaction={'computed_barrier <=': 20})
f_b = Filter(oxidant={'E_H >': 0.5})

step1 = f_a(ds)
step2 = f_b(step1)
```

Chaining is handy when filters are defined in different places or reused across analysis scripts.

## Complex examples

Below is a more involved workflow that demonstrates multiple features together.

```python
# Step 1: base subset
base = Filter(datasets=['Phenols'], reaction={'computed_barrier >': 10})
first = base(ds)

# Step 2: additional constraints on molecules
extra = Filter(
    oxidant={'E_H >=': 1.0},
    substrate={'pKaRH <=': 15},
)
second = extra(first)

# Step 3: custom lambda check
f_final = Filter(func=lambda r: float(r.deltaG0) < 0)
final_subset = f_final(second)
print(f"Final subset contains {len(final_subset)} reactions")
```

## Inspecting filter parameters

All criteria are stored on the dataclass fields and can be examined or modified.

```python
f = Filter(reaction={'computed_barrier >=': 15})
print(f.reaction)
# {'computed_barrier >=': 15}
```

Filters are lightweight, so you can construct them ad hoc without worrying about performance.

## Tips

- Use `Filter()` with no arguments to create a no-op filter.
- Remember that every call returns a new `Dataset`; the original remains unchanged.
- Combine simple filters instead of writing large `query` expressions to keep the logic readable.
- Because filters rely on `Dataset.filter`, they support any column present in the reactions dataframe, including dynamically added ones.


# Variable Metadata

`variable_metadata` in `molecode_utils.var_metadata` is a dictionary mapping variable names to human readable labels and units. It is primarily used by the figure helpers to build axis titles.

```python
from molecode_utils.var_metadata import variable_metadata

info = variable_metadata['deltaG0']
print(info['name'])         # Overall Reaction Free Energy
print(info['latex'])        # LaTeX label
```

Each entry provides:

- `name` – descriptive label
- `latex` – LaTeX formatted label
- `unit_name` / `unit_latex` – textual / LaTeX unit

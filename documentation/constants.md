# Constants Reference

`molecode_utils.constants` defines a few commonly used conversion factors and physical constants. Import them directly when needed.

```python
from molecode_utils.constants import HartreeToKcalMol, VoltsToKcalMol

energy_kcal = 0.5 * HartreeToKcalMol
voltage_kcal = 1.2 * VoltsToKcalMol
```

Available names include:

- `HartreeToKcalMol`
- `HartreeToeVolts`
- `HartreeToJMol`
- `G_solv`
- `R`, `T`, `E_ref`, `F`, `k_B`, `h_Planck`
- `molar_standard_state`
- `VoltsToKcalMol`

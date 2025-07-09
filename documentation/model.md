# Model Cheatsheet

The `molecode_utils.model` module defines an abstract `Model` base class and several Marcus-style implementations (`ModelS`, `ModelM1`–`ModelM4`). Models operate on `Reaction` objects or `Dataset` views.

## Predicting barriers

```python
from molecode_utils.model import ModelM4

m = ModelM4()
rxn_pred = m.predict(reaction)          # single Reaction -> float
subset_pred = m.predict(dataset)        # Dataset -> pandas.Series
```

Each model must implement `_predict_one(self, rxn)` which returns a barrier in kcal/mol. The public `predict` dispatches on input type and also works with any iterable of `Reaction` objects.

## Residuals and evaluation

```python
err = m.residual(reaction)
df = m.evaluate(dataset)    # returns DataFrame with pred/actual/residual
print(df.attrs['MAE'], df.attrs['RMSE'])
```

## Custom models

Subclass `Model` and implement `_predict_one`:

```python
from molecode_utils.model import Model

class MyModel(Model):
    name = "DIY"
    def _predict_one(self, rxn):
        return 0.4 * float(rxn.deltaG0) + 0.2 * float(rxn.RC_formation_energy)
```

`predict`, `residual` and `evaluate` are inherited automatically.

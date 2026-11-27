# TODO after 1.0

## Not in 1.0 (the README Roadmap)

- Hyperparameter search over a whole `Pipeline`: `smallaxe/search/optimize.py` (`run` accepts only a `Model`).
- An `extra_params` pass-through for estimator parameters outside an algorithm's table: `smallaxe/training/algorithm.py`.
- Native CatBoost categorical features (`accepts_raw_categoricals` is `False`): `smallaxe/training/catboost.py`.
- Multilabel classification and quantile regression: `smallaxe/training/algorithm.py` (the task names).
- MLflow integration: `smallaxe/training/mixins/persistence_mixin.py`, `smallaxe/_artifact.py`.
- Spark Connect sessions (refused with `ConfigurationError`): `smallaxe/_fs.py`.
- Multiclass tasks in `AutomatedTraining` (refused with `ValidationError`): `smallaxe/auto/_automated.py`.
- A threshold curve and multiclass ROC curves: `smallaxe/viz/_classification.py`.

## Engineering follow-ups

- Public split utilities; the split logic still lives in the mixin: `smallaxe/training/mixins/validation_mixin.py`.
- Fold the five mixins into the model module (Goals.md, Architecture Deepening 6): `smallaxe/training/base.py`.
- A metric function that returns confusion-matrix counts (only the chart exists): `smallaxe/metrics/classification.py`.
- Decide whether inference without `feature_cols` should have a strict mode: `smallaxe/training/base.py` (`_infer_feature_cols`).
- Check what LightGBM and CatBoost predict for NaN in a feature that had none in training: `examples/databricks_all_algorithms_validation.py`.
- Validate SynapseML's `-spark4.1` and `catboost-spark_4.1_2.13` builds: `smallaxe/training/lightgbm.py`, `smallaxe/training/catboost.py`.
- Python 3.13 and later (`requires-python` stops at 3.12): `pyproject.toml`.
- CI has no job for LightGBM or CatBoost, and does not lint `examples/`: `.github/workflows/ci.yml`.

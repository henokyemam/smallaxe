# Glossary

Domain words used in smallaxe code, tests, and docs. Use these names; do not introduce synonyms.

| Term | Meaning | In code |
|---|---|---|
| **Algorithm** | One of the learning methods smallaxe can train: `random_forest`, `xgboost`, `lightgbm`, `catboost`. An algorithm is described by a declarative record (its param table, Spark estimator and model classes per task, column names, task-fixed params, importance accessor, dependency) plus at most two code hooks (`prepare`, `fit_context`). | `smallaxe/training/algorithm.py` defines `Algorithm`; each algorithm file exports one `ALGORITHM` constant; `algorithm.get(name)` / `algorithm.names()` are the registry. |
| **Param** | One row of an algorithm's strict parameter table: smallaxe name, description, default, Spark estimator name, optional setter, and the tasks it applies to. Users may set only listed params. | `Param` in `algorithm.py`. |
| **Model** | A trainable object with `fit` / `predict` / `save` / `load` / `clone`, bound to one algorithm and one task. Regressors and classifiers are the two kinds. | `Regressor`, `Classifier` in `smallaxe/training/base.py`. The eight 0.8.x class names (`XGBoostRegressor`, ...) are deprecated aliases. |
| **Task** | What the label represents: `simple_regression`, `binary`, or `multiclass`. Determines the estimator class, task-fixed params, metrics, and whether `predict_proba` exists. | `BaseModel.REGRESSION_TASKS` / `CLASSIFICATION_TASKS`; persisted in `metadata.json` as `task`. |
| **Factory** | The documented way to get a model: `Regressors.xgboost(...)`, `Classifiers.catboost(task=...)`. A factory returns a plain `Regressor` / `Classifier`. | `smallaxe/training/regressors.py`, `classifiers.py`. |
| **Pipeline** | Ordered preprocessing steps followed by one model, trained and applied as one unit. | `smallaxe/pipeline/pipeline.py`. |
| **Step** | One stage of a pipeline: `Imputer`, `Scaler`, `Encoder`, or a model. | `smallaxe/preprocessing/`. |
| **Validation** | How `fit` estimates generalisation: `none`, `train_test`, or `kfold` (stratified for classification by default). | `ValidationMixin`. |
| **Artifact** | A saved directory: the Spark model (if any) plus `metadata.json`, written last. Its header (`format_version`, `kind`, `smallaxe_version`) says what it is and which format it uses; model artifacts have kind `model`. Model artifacts written by 0.10 and earlier have no header (0.8.x ones record the old class name, 0.9 and 0.10 ones `algorithm` and `task`) and still load. | `smallaxe/_artifact.py`, `PersistenceMixin.save`, `Model.load` / `smallaxe.load_model`, `smallaxe/_fs.py`. |

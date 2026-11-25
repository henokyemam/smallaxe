"""Pipeline: preprocessing steps and an optional model, fitted, applied and saved as one unit.

A step is one of four kinds, decided by what it is: an ``Imputer``, a ``Scaler``, an
``Encoder`` or a smallaxe model (``"imputer"``, ``"scaler"``, ``"encoder"``, ``"model"``,
the kinds their own artifacts record). Other objects work as steps too, duck-typed: a
preprocessing step is anything with ``fit`` and ``transform`` and no ``predict``, a model
anything with ``fit`` and ``predict``. Only the four kinds can be saved.

The pipeline owns its columns. Each preprocessing step receives the column lists given
to :meth:`Pipeline.fit`, and the model is fitted on exactly ``numerical_cols`` plus the
Encoder's output columns; every other column of the frame passes through untouched.

A pipeline artifact is a directory holding ``metadata.json`` (kind ``"pipeline"``,
written last through :mod:`smallaxe._artifact`) and each saved step's own artifact under
``steps/<index>_<name>/``. All IO goes through :mod:`smallaxe._fs`, so local paths,
``dbfs:/``, ``s3://`` and ``hdfs://`` all work. Nothing is pickled.
"""

import io
import pickle
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast

from pyspark.sql import DataFrame

from smallaxe import _artifact, _deprecation, _fs
from smallaxe.exceptions import ModelNotFittedError, PreprocessingError, ValidationError

ARTIFACT_KIND = "pipeline"
STEPS_DIR = "steps"

IMPUTER = "imputer"
SCALER = "scaler"
ENCODER = "encoder"
MODEL = "model"
STEP_KINDS = (IMPUTER, SCALER, ENCODER, MODEL)
PREPROCESSING_KINDS = (IMPUTER, SCALER, ENCODER)

# Model.predict / predict_proba output column defaults.
DEFAULT_PREDICTION_COL = "predict_label"
DEFAULT_PROBABILITY_COL = "probability"

# smallaxe 0.10 Pipeline.save recorded the class name of each step in ``step_types``.
# Its Encoder had no save(), so it was pickled: the one pickle a legacy pipeline may hold.
_LEGACY_ENCODER_TYPE = "Encoder"
_LEGACY_ENCODER_GLOBAL = ("smallaxe.preprocessing.encoder", "Encoder")

_UNSAFE_DIR_CHARS = re.compile(r"[^A-Za-z0-9_.-]")


def step_kind(step: Any) -> Optional[str]:
    """The kind of a step: ``"imputer"``, ``"scaler"``, ``"encoder"``, ``"model"`` or None.

    ``"model"`` is a smallaxe :class:`~smallaxe.training.base.Model` (a ``Regressor`` or
    ``Classifier``); None is a duck-typed step, which cannot be saved.
    """
    from smallaxe.preprocessing import Encoder, Imputer, Scaler
    from smallaxe.training.base import Model

    if isinstance(step, Imputer):
        return IMPUTER
    if isinstance(step, Scaler):
        return SCALER
    if isinstance(step, Encoder):
        return ENCODER
    if isinstance(step, Model):
        return MODEL
    return None


def _is_model_step(step: Any) -> bool:
    """Whether ``step`` is a model: a smallaxe model, or anything with fit and predict."""
    from smallaxe.training.base import BaseModel

    if isinstance(step, BaseModel):
        return True
    return hasattr(step, "fit") and hasattr(step, "predict")


def _is_preprocessing_step(step: Any) -> bool:
    """Whether ``step`` is a preprocessing step: a smallaxe one, or fit + transform, no predict."""
    if step_kind(step) in PREPROCESSING_KINDS:
        return True
    return hasattr(step, "fit") and hasattr(step, "transform") and not hasattr(step, "predict")


def _step_dir(index: int, name: str) -> str:
    """The step's directory relative to the pipeline artifact: ``steps/<index>_<name>``."""
    return _fs.join(STEPS_DIR, f"{index}_{_UNSAFE_DIR_CHARS.sub('_', name)}")


def _skipped_params(kind: str, step: Any) -> Dict[str, Any]:
    """The constructor arguments of a skipped (unfitted) preprocessing step."""
    if kind == IMPUTER:
        return {
            "numerical_strategy": step._numerical_strategy,
            "categorical_strategy": step._categorical_strategy,
        }
    if kind == SCALER:
        return {"method": step.method}
    return {
        "method": step.method,
        "max_categories": step.max_categories,
        "handle_rare": step.handle_rare,
    }


def _loaders() -> Dict[str, Callable[[str], Any]]:
    """Step kind -> the function that loads a step artifact of that kind."""
    from smallaxe import load_model
    from smallaxe.preprocessing import Encoder, Imputer, Scaler

    return {IMPUTER: Imputer.load, SCALER: Scaler.load, ENCODER: Encoder.load, MODEL: load_model}


def _constructors() -> Dict[str, Callable[..., Any]]:
    """Preprocessing kind -> its class, to rebuild a skipped step."""
    from smallaxe.preprocessing import Encoder, Imputer, Scaler

    return {IMPUTER: Imputer, SCALER: Scaler, ENCODER: Encoder}


class _LegacyEncoderUnpickler(pickle.Unpickler):
    """Unpickles a smallaxe 0.10 Encoder and nothing else: no other global is resolved."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == _LEGACY_ENCODER_GLOBAL:
            from smallaxe.preprocessing import Encoder

            return Encoder
        raise pickle.UnpicklingError(
            f"global {module}.{name} is not allowed; only {'.'.join(_LEGACY_ENCODER_GLOBAL)} is"
        )


class Pipeline:
    """Chain preprocessing steps and optionally one model, and fit, apply and save them.

    ``fit`` hands each step the column lists it is given: an ``Imputer`` gets
    ``numerical_cols`` and ``categorical_cols``, a ``Scaler`` ``numerical_cols``, an
    ``Encoder`` ``categorical_cols``, and a duck-typed preprocessing step both, as
    ``fit(dataframe, numerical_cols=..., categorical_cols=...)``. A smallaxe step left with no
    columns is skipped: it stays unfitted, ``transform`` passes the data through it, and
    it is saved and restored as skipped (see :attr:`skipped_steps`).

    The model is fitted on exactly ``numerical_cols`` plus the Encoder's
    :attr:`~smallaxe.preprocessing.Encoder.output_cols` (the raw ``categorical_cols``
    when there is no Encoder and the model needs none); columns not listed never reach
    it, and the label is never a feature. With neither list the model infers its
    features as ``Model.fit`` does without ``feature_cols``. The chosen columns are
    :attr:`feature_cols`.

    A smallaxe model whose algorithm does not accept raw categorical columns (every
    algorithm today) needs an ``Encoder`` in the pipeline whenever ``categorical_cols``
    are given; ``fit`` raises ``PreprocessingError`` otherwise.

    Args:
        steps: A list of ``(name, step)`` tuples: preprocessing steps first, then at
            most one model. Step names must be unique non-empty strings.

    Raises:
        ValidationError: If steps is empty or malformed, names are not unique, a step is
            neither a preprocessing step nor a model, there is more than one model, or a
            preprocessing step comes after the model.

    Example:
        >>> from smallaxe.pipeline import Pipeline
        >>> from smallaxe.preprocessing import Encoder, Imputer, Scaler
        >>> from smallaxe.training import Classifiers
        >>> pipeline = Pipeline([
        ...     ("imputer", Imputer(numerical_strategy="median",
        ...                         categorical_strategy="most_frequent")),
        ...     ("scaler", Scaler()),
        ...     ("encoder", Encoder()),
        ...     ("model", Classifiers.random_forest()),
        ... ])
        >>> pipeline.fit(dataframe, label_col="label", numerical_cols=["age"],
        ...              categorical_cols=["city"])
        >>> pipeline.save("/path/to/pipeline")
        >>> predictions = Pipeline.load("/path/to/pipeline").predict(new_df)
    """

    def __init__(self, steps: List[Tuple[str, Any]]) -> None:
        self._validate_steps(steps)
        self._steps: List[Tuple[str, Any]] = list(steps)
        self._is_fitted = False
        self._has_model = any(_is_model_step(step) for _, step in self._steps)
        self._numerical_cols: Optional[List[str]] = None
        self._categorical_cols: Optional[List[str]] = None
        self._label_col: Optional[str] = None
        self._feature_cols: Optional[List[str]] = None
        self._skipped: List[str] = []

    # --- Validation -----------------------------------------------------------------

    @staticmethod
    def _validate_steps(steps: List[Tuple[str, Any]]) -> None:
        """Check the steps' shape, names, kinds and order.

        Raises:
            ValidationError: As described in the class docstring.
        """
        if not steps:
            raise ValidationError(message="Pipeline steps cannot be empty")
        if not isinstance(steps, list):
            raise ValidationError(message="steps must be a list of (name, step) tuples")

        for entry in steps:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise ValidationError(message="Each step must be a (name, step) tuple")
            name, _ = entry
            if not isinstance(name, str):
                raise ValidationError(
                    message=f"Step name must be a string, got {type(name).__name__}"
                )
            if not name:
                raise ValidationError(message="Step name cannot be empty")

        names = [name for name, _ in steps]
        if len(names) != len(set(names)):
            duplicates = {name for name in names if names.count(name) > 1}
            raise ValidationError(
                message=f"Step names must be unique. Duplicates found: {duplicates}"
            )

        model_name: Optional[str] = None
        for name, step in steps:
            if _is_model_step(step):
                if model_name is not None:
                    raise ValidationError(
                        message=(
                            f"A pipeline holds at most one model; '{model_name}' and '{name}' "
                            "are both models."
                        )
                    )
                model_name = name
            elif _is_preprocessing_step(step):
                if model_name is not None:
                    raise ValidationError(
                        message=(
                            f"Invalid step order: preprocessing step '{name}' "
                            f"cannot appear after model step '{model_name}'. "
                            "All preprocessing steps must come before model steps."
                        )
                    )
            else:
                raise ValidationError(
                    message=(
                        f"Step '{name}' is neither a preprocessing step (fit and transform) "
                        "nor a model (fit and predict)."
                    )
                )

    def _check_encoder_requirement(self, categorical_cols: List[str]) -> None:
        """Raise PreprocessingError if the model needs an Encoder the pipeline lacks.

        Raises:
            PreprocessingError: If ``categorical_cols`` are given, there is no Encoder,
                and the model is a smallaxe model whose algorithm does not accept raw
                categorical columns.
        """
        if not categorical_cols:
            return
        if any(step_kind(step) == ENCODER for _, step in self._steps):
            return
        for _, step in self._steps:
            if step_kind(step) == MODEL and not step.algorithm.accepts_raw_categoricals:
                raise PreprocessingError(algorithm=step.algorithm.name, missing_step="Encoder")

    # --- Properties -----------------------------------------------------------------

    @property
    def steps(self) -> List[Tuple[str, Any]]:
        """A copy of the ``(name, step)`` list."""
        return list(self._steps)

    @property
    def named_steps(self) -> Dict[str, Any]:
        """The steps as a dictionary from name to step."""
        return {name: step for name, step in self._steps}

    @property
    def feature_cols(self) -> Optional[List[str]]:
        """The columns the model was fitted on, or None before fit or without a model."""
        return list(self._feature_cols) if self._feature_cols is not None else None

    @property
    def skipped_steps(self) -> List[str]:
        """Names of the steps the last fit skipped because they received no columns."""
        return list(self._skipped)

    # --- Fit and apply --------------------------------------------------------------

    @_deprecation.df_keyword
    def fit(
        self,
        dataframe: DataFrame,
        label_col: Optional[str] = None,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> "Pipeline":
        """Fit every step in order, each on the output of the steps before it.

        Args:
            dataframe: The PySpark DataFrame to fit on.
            label_col: The label column; required when the pipeline has a model.
            numerical_cols: Numerical columns: imputed, scaled, and model features.
            categorical_cols: Categorical columns: imputed, encoded, and (encoded)
                model features.

        Returns:
            self, for method chaining.

        Raises:
            ValidationError: If ``label_col`` is missing when the pipeline has a model,
                or is listed in ``numerical_cols`` / ``categorical_cols``.
            PreprocessingError: If ``categorical_cols`` are given and the model needs
                an Encoder the pipeline does not have.

        Any error a step's own ``fit`` raises propagates unchanged.
        """
        self._is_fitted = False
        numerical = list(numerical_cols or [])
        categorical = list(categorical_cols or [])

        if self._has_model and label_col is None:
            raise ValidationError(message="label_col is required when pipeline contains a model")
        if label_col is not None and label_col in numerical + categorical:
            raise ValidationError(
                message=(
                    f"label_col '{label_col}' is also listed in numerical_cols or "
                    "categorical_cols. The label is never a feature; remove it from those lists."
                )
            )
        self._check_encoder_requirement(categorical)

        skipped: List[str] = []
        encoder_outputs: Optional[List[str]] = None
        model: Any = None
        current = dataframe
        for name, step in self._steps:
            if _is_model_step(step):
                model = step
                break
            kind = step_kind(step)
            if kind == IMPUTER:
                if not numerical and not categorical:
                    skipped.append(name)
                    continue
                step.fit(
                    current,
                    numerical_cols=numerical or None,
                    categorical_cols=categorical or None,
                )
            elif kind == SCALER:
                if not numerical:
                    skipped.append(name)
                    continue
                step.fit(current, numerical_cols=numerical)
            elif kind == ENCODER:
                if not categorical:
                    skipped.append(name)
                    continue
                step.fit(current, categorical_cols=categorical)
                encoder_outputs = step.output_cols
            else:
                step.fit(current, numerical_cols=numerical_cols, categorical_cols=categorical_cols)
            current = step.transform(current)

        feature_cols: Optional[List[str]] = None
        if model is not None:
            if numerical or categorical:
                encoded = encoder_outputs if encoder_outputs is not None else categorical
                feature_cols = numerical + list(encoded)
            feature_cols = self._fit_model(model, current, str(label_col), feature_cols)

        self._numerical_cols = list(numerical_cols) if numerical_cols is not None else None
        self._categorical_cols = list(categorical_cols) if categorical_cols is not None else None
        self._label_col = label_col
        self._feature_cols = feature_cols
        self._skipped = skipped
        self._is_fitted = True
        return self

    @staticmethod
    def _fit_model(
        model: Any, dataframe: DataFrame, label_col: str, feature_cols: Optional[List[str]]
    ) -> List[str]:
        """Fit the model step and return the feature columns it was fitted on.

        With ``feature_cols`` None, a smallaxe model infers its features in its own
        ``fit``; a duck-typed model gets the columns the same inference picks.
        """
        from smallaxe.training.base import BaseModel, infer_feature_cols, warn_ignored_cols

        if isinstance(model, BaseModel):
            model.fit(dataframe, label_col=label_col, feature_cols=feature_cols)
            return list(model._feature_cols)
        if feature_cols is None:
            feature_cols, ignored = infer_feature_cols(dataframe, label_col, [])
            warn_ignored_cols(feature_cols, ignored)
        model.fit(dataframe, label_col=label_col, feature_cols=feature_cols)
        return list(feature_cols)

    @_deprecation.df_keyword
    def transform(self, dataframe: DataFrame) -> DataFrame:
        """Apply every fitted preprocessing step in order; skipped steps pass data through.

        Args:
            dataframe: The PySpark DataFrame to transform.

        Returns:
            The transformed DataFrame. Columns no step was fitted on are unchanged.

        Raises:
            ModelNotFittedError: If called before fit.
        """
        if not self._is_fitted:
            raise ModelNotFittedError("Pipeline has not been fitted. Call fit() first.")

        current = dataframe
        for name, step in self._steps:
            if _is_model_step(step):
                break
            if name in self._skipped:
                continue
            current = step.transform(current)
        return current

    @_deprecation.df_keyword
    def fit_transform(
        self,
        dataframe: DataFrame,
        label_col: Optional[str] = None,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> DataFrame:
        """Fit the pipeline, then transform ``dataframe`` with it.

        Args:
            dataframe: The PySpark DataFrame to fit on and transform.
            label_col: The label column; required when the pipeline has a model.
            numerical_cols: Numerical columns (see :meth:`fit`).
            categorical_cols: Categorical columns (see :meth:`fit`).

        Returns:
            The transformed DataFrame.
        """
        self.fit(
            dataframe,
            label_col=label_col,
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
        )
        return self.transform(dataframe)

    def _fitted_model(self, method: str) -> Any:
        """The model step, for ``method``.

        Raises:
            ModelNotFittedError: If the pipeline has not been fitted.
            ValidationError: If the pipeline has no model.
        """
        if not self._is_fitted:
            raise ModelNotFittedError("Pipeline has not been fitted. Call fit() first.")
        for _, step in self._steps:
            if _is_model_step(step):
                return step
        raise ValidationError(
            message=(
                f"Pipeline does not contain a model. {method}() is only available for "
                "pipelines with models."
            )
        )

    @_deprecation.df_keyword
    def predict(self, dataframe: DataFrame, output_col: str = DEFAULT_PREDICTION_COL) -> DataFrame:
        """Apply the preprocessing steps, then predict with the model.

        Args:
            dataframe: The PySpark DataFrame to predict on. Columns the pipeline was not fitted
                on are passed through.
            output_col: Name of the prediction column, passed to the model's
                ``predict``. Default is 'predict_label', as for ``Model.predict``. A
                duck-typed model receives it only when it is not the default.

        Returns:
            The transformed DataFrame with the prediction column added.

        Raises:
            ModelNotFittedError: If called before fit.
            ValidationError: If the pipeline does not contain a model.
        """
        from smallaxe.training.base import BaseModel

        model = self._fitted_model("predict")
        current = self.transform(dataframe)
        if isinstance(model, BaseModel) or output_col != DEFAULT_PREDICTION_COL:
            return cast(DataFrame, model.predict(current, output_col=output_col))
        return cast(DataFrame, model.predict(current))

    @_deprecation.df_keyword
    def predict_proba(
        self, dataframe: DataFrame, output_col: str = DEFAULT_PROBABILITY_COL
    ) -> DataFrame:
        """Apply the preprocessing steps, then predict class probabilities with the model.

        Args:
            dataframe: The PySpark DataFrame to predict on.
            output_col: Name of the probability column, passed to the model's
                ``predict_proba``. Default is 'probability', as for
                ``Classifier.predict_proba``.

        Returns:
            The transformed DataFrame with the probability column added.

        Raises:
            ModelNotFittedError: If called before fit.
            ValidationError: If the pipeline has no model, or its model is a regressor
                or has no ``predict_proba``.
        """
        from smallaxe.training.base import BaseClassifier, BaseModel

        model = self._fitted_model("predict_proba")
        if isinstance(model, BaseModel):
            if not isinstance(model, BaseClassifier):
                raise ValidationError(
                    message=(
                        "predict_proba() needs a classifier; this pipeline's model is a "
                        f"{model.task} model."
                    )
                )
        elif not hasattr(model, "predict_proba"):
            raise ValidationError(message="predict_proba() needs a model with predict_proba.")
        current = self.transform(dataframe)
        if isinstance(model, BaseModel) or output_col != DEFAULT_PROBABILITY_COL:
            return cast(DataFrame, model.predict_proba(current, output_col=output_col))
        return cast(DataFrame, model.predict_proba(current))

    @_deprecation.df_keyword
    def evaluate(self, dataframe: DataFrame, label_col: Optional[str] = None) -> Dict[str, Any]:
        """Apply the preprocessing steps, then score the model as ``Model.evaluate`` does.

        Args:
            dataframe: The PySpark DataFrame to score, holding the label.
            label_col: The label column. Defaults to the label the pipeline was fitted
                with.

        Returns:
            Dictionary mapping metric names to values; undefined metrics are None.

        Raises:
            ModelNotFittedError: If called before fit.
            ValidationError: If the pipeline has no model, or a duck-typed model has no
                ``evaluate``.
            ColumnNotFoundError: If the label or a feature column is missing.
        """
        model = self._fitted_model("evaluate")
        if not hasattr(model, "evaluate"):
            raise ValidationError(message="evaluate() needs a model with evaluate.")
        return dict(
            model.evaluate(self.transform(dataframe), label_col=label_col or self._label_col)
        )

    # --- Persistence ----------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the fitted pipeline as a directory at ``path``.

        Each step that was fitted is saved by its own ``save`` into
        ``<path>/steps/<index>_<name>/``; skipped steps are recorded, not saved. Then
        ``<path>/metadata.json`` is written, with the ordered steps (name, kind,
        directory, skipped), ``numerical_cols``, ``categorical_cols``, ``label_col``,
        ``feature_cols`` and ``has_model``. Saving over an existing pipeline first
        removes its ``metadata.json`` and its step directories (in either this layout
        or smallaxe 0.10's), so nothing of it is left. All IO goes through the Hadoop
        FileSystem layer, so ``dbfs:/``, ``s3://`` and ``hdfs://`` paths work.

        Args:
            path: Directory path to save the pipeline to.

        Raises:
            ModelNotFittedError: If called before fit.
            ValidationError: If the path is empty, or a step is not an Imputer, Scaler,
                Encoder or smallaxe model (nothing is written then).
        """
        if not self._is_fitted:
            raise ModelNotFittedError("Pipeline has not been fitted. Call fit() before saving.")
        if not path:
            raise ValidationError("Path cannot be empty.")

        entries: List[Dict[str, Any]] = []
        for index, (name, step) in enumerate(self._steps):
            kind = step_kind(step)
            if kind is None:
                raise ValidationError(
                    f"Cannot save pipeline: step '{name}' ({type(step).__name__}) is not an "
                    "Imputer, Scaler, Encoder or smallaxe model, so it has no saved form. "
                    "Nothing was written."
                )
            entry: Dict[str, Any] = {
                "name": name,
                "kind": kind,
                "path": _step_dir(index, name),
                "skipped": name in self._skipped,
            }
            if entry["skipped"]:
                entry["params"] = _skipped_params(kind, step)
            entries.append(entry)

        self._remove_saved_pipeline(path)
        for entry, (_, step) in zip(entries, self._steps):
            if not entry["skipped"]:
                step.save(_fs.join(path, entry["path"]))
        _artifact.write_metadata(
            path,
            ARTIFACT_KIND,
            {
                "steps": entries,
                "numerical_cols": self._numerical_cols,
                "categorical_cols": self._categorical_cols,
                "label_col": self._label_col,
                "feature_cols": self._feature_cols,
                "has_model": self._has_model,
            },
        )

    @staticmethod
    def _remove_saved_pipeline(path: str) -> None:
        """Delete what a pipeline saved at ``path`` wrote, in this layout or 0.10's.

        Only the pipeline's own files are removed: ``metadata.json``, ``steps/``, and
        for a 0.10 pipeline its ``step_<i>_<name>`` directories and pickles.
        """
        metadata_path = _fs.join(path, _artifact.METADATA_FILE)
        if not _fs.exists(metadata_path):
            return
        legacy: List[str] = []
        try:
            existing = _fs.read_json(metadata_path)
        except (ValueError,) + _fs.jvm_error_types():
            existing = None
        if isinstance(existing, dict) and _artifact.is_unversioned(existing):
            names = existing.get("step_names")
            if isinstance(names, list):
                for index, name in enumerate(names):
                    if isinstance(name, str) and "/" not in name and "\\" not in name:
                        step_path = _fs.join(path, f"step_{index}_{name}")
                        legacy.extend([step_path, step_path + ".pkl"])
        _fs.delete(metadata_path)
        _fs.delete(_fs.join(path, STEPS_DIR), recursive=True)
        for legacy_path in legacy:
            _fs.delete(legacy_path, recursive=True)

    @classmethod
    def load(cls, path: str) -> "Pipeline":
        """Load a pipeline saved by :meth:`save`, or a preprocessing pipeline of 0.10.

        Each step is loaded by its kind: ``Imputer.load``, ``Scaler.load``,
        ``Encoder.load``, ``smallaxe.load_model``. A pipeline saved by smallaxe 0.10 (an
        unversioned ``metadata.json`` with ``step_names`` / ``step_types``) loads when
        its steps are an Imputer, a Scaler and an Encoder: the Encoder that 0.10 pickled
        is the one pickle read, through an unpickler that resolves nothing but the
        Encoder class. Any other pickle in it is refused.

        Args:
            path: The artifact directory.

        Returns:
            The fitted pipeline.

        Raises:
            ValidationError: If the path or its ``metadata.json`` does not exist, it holds
                another kind of artifact or was written by a newer smallaxe, a step
                directory is missing, the metadata is incomplete or corrupted, or a 0.10
                pipeline holds a pickle other than its Encoder or a step that is not an
                Imputer, Scaler or Encoder.
            DependencyError: If the model step's algorithm is not installed.
        """
        metadata = _artifact.read_metadata(path, ARTIFACT_KIND, allow_unversioned=True)
        if _artifact.is_unversioned(metadata):
            return cls._load_legacy(path, metadata)
        return cls._load_current(path, metadata)

    @classmethod
    def _load_current(cls, path: str, metadata: Dict[str, Any]) -> "Pipeline":
        """Load a pipeline written in the current layout."""
        keys = [
            "steps",
            "numerical_cols",
            "categorical_cols",
            "label_col",
            "feature_cols",
            "has_model",
        ]
        missing = [key for key in keys if key not in metadata]
        if missing:
            raise ValidationError(
                f"Cannot load pipeline: the metadata at {path!r} is missing {missing}; the "
                "artifact is incomplete or corrupted."
            )
        entries = metadata["steps"]
        if not isinstance(entries, list) or not entries:
            raise ValidationError(
                f"Cannot load pipeline: 'steps' at {path!r} must be a non-empty list, got "
                f"{entries!r}; the artifact is corrupted."
            )

        loaders = _loaders()
        steps: List[Tuple[str, Any]] = []
        skipped: List[str] = []
        for entry in entries:
            name, kind, step_dir, is_skipped = _read_entry(path, entry)
            if is_skipped:
                steps.append((name, _rebuild_skipped(path, name, kind, entry.get("params"))))
                skipped.append(name)
                continue
            step_path = _fs.join(path, step_dir)
            if not _fs.exists(step_path):
                raise ValidationError(
                    f"Cannot load pipeline: step '{name}' ({kind}) is missing; its directory "
                    f"{step_path!r} does not exist. The artifact is incomplete."
                )
            steps.append((name, loaders[kind](step_path)))

        pipeline = cls._restore(
            path,
            steps,
            skipped,
            numerical_cols=_optional_columns(path, metadata, "numerical_cols"),
            categorical_cols=_optional_columns(path, metadata, "categorical_cols"),
            label_col=metadata["label_col"],
            feature_cols=_optional_columns(path, metadata, "feature_cols"),
        )
        for _, step in pipeline._steps:
            if step_kind(step) == MODEL and list(step._feature_cols) != pipeline._feature_cols:
                raise ValidationError(
                    f"Cannot load pipeline: 'feature_cols' at {path!r} "
                    f"({pipeline._feature_cols!r}) differ from the model's "
                    f"({step._feature_cols!r}); the artifact is corrupted."
                )
        return pipeline

    @classmethod
    def _load_legacy(cls, path: str, metadata: Dict[str, Any]) -> "Pipeline":
        """Load a preprocessing pipeline saved by smallaxe 0.10 (unversioned, local)."""
        names = metadata.get("step_names")
        types = metadata.get("step_types")
        if names is None and types is None:
            raise ValidationError(
                f"Cannot load pipeline: {path!r} holds an unversioned artifact without "
                "'step_names'; it looks like a model saved by smallaxe 0.10 or earlier. Load "
                "it with smallaxe.load_model."
            )
        if (
            not isinstance(names, list)
            or not isinstance(types, list)
            or len(names) != len(types)
            or not all(isinstance(value, str) and value for value in names + types)
            or any("/" in name or "\\" in name for name in names)
        ):
            raise ValidationError(
                f"Cannot load pipeline: 'step_names' / 'step_types' at {path!r} are not two "
                "lists of names of the same length; the artifact is corrupted."
            )

        from smallaxe.preprocessing import Encoder, Imputer, Scaler

        # A step's own pickle (step_<i>_<name>.pkl) is judged with its step below.
        _refuse_pickles(
            path, path, {f"step_{index}_{name}.pkl" for index, name in enumerate(names)}
        )

        steps: List[Tuple[str, Any]] = []
        skipped: List[str] = []
        for index, (name, type_name) in enumerate(zip(names, types)):
            step_path = _fs.join(path, f"step_{index}_{name}")
            pickle_path = step_path + ".pkl"
            if _fs.exists(step_path):
                _refuse_pickles(path, step_path, set())
            if _fs.exists(pickle_path):
                if type_name != _LEGACY_ENCODER_TYPE:
                    raise ValidationError(
                        f"Cannot load pipeline: step '{name}' ({type_name}) of the smallaxe "
                        f"0.10 pipeline at {path!r} was pickled ({pickle_path!r}). Pickled "
                        "steps are no longer loaded, because unpickling can run arbitrary "
                        "code; only the Encoder that smallaxe 0.10 pickled itself is read. "
                        "Refit the pipeline and save it again."
                    )
                step: Any = _unpickle_legacy_encoder(pickle_path)
                if not step._is_fitted:
                    skipped.append(name)
            elif type_name == "Imputer":
                step = Imputer.load(step_path)
            elif type_name == "Scaler":
                step = Scaler.load(step_path)
            elif type_name == _LEGACY_ENCODER_TYPE:
                step = Encoder.load(step_path)
            else:
                raise ValidationError(
                    f"Cannot load pipeline: step '{name}' of the smallaxe 0.10 pipeline at "
                    f"{path!r} has type {type_name!r}; only its Imputer, Scaler and Encoder "
                    "steps can be loaded. Refit the pipeline and save it again."
                )
            steps.append((name, step))

        return cls._restore(
            path,
            steps,
            skipped,
            numerical_cols=_optional_columns(path, metadata, "numerical_cols"),
            categorical_cols=_optional_columns(path, metadata, "categorical_cols"),
            label_col=metadata.get("label_col"),
            feature_cols=None,
        )

    @classmethod
    def _restore(
        cls,
        path: str,
        steps: List[Tuple[str, Any]],
        skipped: List[str],
        numerical_cols: Optional[List[str]],
        categorical_cols: Optional[List[str]],
        label_col: Any,
        feature_cols: Optional[List[str]],
    ) -> "Pipeline":
        """Build a fitted pipeline from loaded steps and its recorded state."""
        if label_col is not None and not isinstance(label_col, str):
            raise ValidationError(
                f"Cannot load pipeline: 'label_col' at {path!r} must be a string or null, got "
                f"{label_col!r}; the artifact is corrupted."
            )
        try:
            pipeline = cls(steps)
        except ValidationError as exc:
            raise ValidationError(
                f"Cannot load pipeline: the steps at {path!r} do not form a pipeline ({exc})."
            ) from exc
        pipeline._numerical_cols = numerical_cols
        pipeline._categorical_cols = categorical_cols
        pipeline._label_col = label_col
        pipeline._feature_cols = feature_cols
        pipeline._skipped = skipped
        pipeline._is_fitted = True
        return pipeline

    # --- Representation -------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a string representation of the pipeline."""
        step_strs = [f"('{name}', {type(step).__name__})" for name, step in self._steps]
        fitted_str = "fitted" if self._is_fitted else "not fitted"
        return f"Pipeline([{', '.join(step_strs)}], {fitted_str})"

    def __len__(self) -> int:
        """Return the number of steps in the pipeline."""
        return len(self._steps)

    def __getitem__(self, key: str) -> Any:
        """Get a step by name.

        Args:
            key: The step name.

        Returns:
            The step object.

        Raises:
            KeyError: If the step name is not found.
        """
        for name, step in self._steps:
            if name == key:
                return step
        raise KeyError(f"Step '{key}' not found in pipeline")


def _read_entry(path: str, entry: Any) -> Tuple[str, str, str, bool]:
    """Check one ``steps`` entry of the metadata: ``(name, kind, directory, skipped)``.

    The directory must be a relative path inside ``steps/``, so a corrupted or edited
    artifact cannot point a loader outside the pipeline directory.

    Raises:
        ValidationError: If the entry is malformed.
    """
    if not isinstance(entry, dict):
        raise ValidationError(
            f"Cannot load pipeline: a 'steps' entry at {path!r} is not an object ({entry!r}); "
            "the artifact is corrupted."
        )
    name = entry.get("name")
    kind = entry.get("kind")
    step_dir = entry.get("path")
    skipped = entry.get("skipped")
    if not isinstance(name, str) or not name:
        raise ValidationError(
            f"Cannot load pipeline: a step at {path!r} has no name ({entry!r}); the artifact "
            "is corrupted."
        )
    if not isinstance(kind, str) or kind not in STEP_KINDS:
        raise ValidationError(
            f"Cannot load pipeline: step '{name}' at {path!r} has kind {kind!r}; this smallaxe "
            f"knows {list(STEP_KINDS)}. It may have been written by a newer smallaxe."
        )
    if not isinstance(skipped, bool) or (skipped and kind == MODEL):
        raise ValidationError(
            f"Cannot load pipeline: step '{name}' at {path!r} has an invalid 'skipped' "
            f"({skipped!r}); the artifact is corrupted."
        )
    parts = step_dir.split("/") if isinstance(step_dir, str) else []
    if (
        not isinstance(step_dir, str)
        or len(parts) != 2
        or parts[0] != STEPS_DIR
        or parts[1] in ("", ".", "..")
    ):
        raise ValidationError(
            f"Cannot load pipeline: step '{name}' at {path!r} has the directory {step_dir!r}, "
            f"which is not inside {STEPS_DIR}/; the artifact is corrupted."
        )
    return name, kind, step_dir, skipped


def _rebuild_skipped(path: str, name: str, kind: str, params: Any) -> Any:
    """An unfitted step of ``kind`` built from its recorded constructor arguments."""
    if not isinstance(params, dict):
        raise ValidationError(
            f"Cannot load pipeline: skipped step '{name}' at {path!r} has no 'params'; the "
            "artifact is corrupted."
        )
    try:
        return _constructors()[kind](**params)
    except (TypeError, ValidationError) as exc:
        raise ValidationError(
            f"Cannot load pipeline: skipped step '{name}' at {path!r} has invalid 'params' "
            f"({exc}); the artifact is corrupted."
        ) from exc


def _optional_columns(path: str, metadata: Dict[str, Any], key: str) -> Optional[List[str]]:
    """``metadata[key]`` as a list of column names or None, else ValidationError."""
    value = metadata.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(col, str) for col in value):
        raise ValidationError(
            f"Cannot load pipeline: {key!r} at {path!r} must be a list of column names or "
            f"null, got {value!r}; the artifact is corrupted."
        )
    return list(value)


def _refuse_pickles(path: str, directory: str, allowed: Set[str]) -> None:
    """Raise ValidationError if ``directory`` holds a ``.pkl`` file not in ``allowed``."""
    for entry in sorted(_fs.list_names(directory)):
        if entry.endswith(".pkl") and entry not in allowed:
            raise ValidationError(
                f"Cannot load pipeline: {_fs.join(directory, entry)!r} in the smallaxe 0.10 "
                f"pipeline at {path!r} is a pickle. Pickled steps are no longer loaded, "
                "because unpickling can run arbitrary code; only the Encoder that smallaxe "
                "0.10 pickled itself (step_<i>_<name>.pkl of an Encoder step) is read. "
                "Remove the file, or refit the pipeline and save it again."
            )


def _unpickle_legacy_encoder(pickle_path: str) -> Any:
    """Read the Encoder that smallaxe 0.10's Pipeline.save pickled.

    Raises:
        ValidationError: If the file references anything but the Encoder class, cannot
            be unpickled, or does not hold an Encoder.
    """
    from smallaxe.preprocessing import Encoder

    data = _fs.read_bytes(pickle_path)
    try:
        step = _LegacyEncoderUnpickler(io.BytesIO(data)).load()
    except (
        pickle.UnpicklingError,
        EOFError,
        AttributeError,
        ImportError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise ValidationError(
            f"Cannot load pipeline: {pickle_path!r} is not a pickled smallaxe 0.10 Encoder "
            f"({exc})."
        ) from exc
    if not isinstance(step, Encoder):
        raise ValidationError(
            f"Cannot load pipeline: {pickle_path!r} holds a {type(step).__name__}, not an "
            "Encoder."
        )
    return step

"""Pipeline class for chaining preprocessing and model steps."""

import json
import os
import pickle
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple

from pyspark.sql import DataFrame

from smallaxe.exceptions import ModelNotFittedError, ValidationError


class Pipeline:
    """A pipeline for chaining preprocessing steps and optionally a model.

    Pipeline allows you to chain multiple preprocessing steps (like Imputer,
    Scaler, Encoder) together, and optionally include a model at the end.
    This enables fit/transform operations to be applied sequentially.

    Parameters
    ----------
    steps : List[Tuple[str, Any]]
        A list of (name, step) tuples where each step is a preprocessing
        component or model. Step names must be unique.

    Examples
    --------
    >>> from smallaxe.pipeline import Pipeline
    >>> from smallaxe.preprocessing import Imputer, Scaler, Encoder
    >>>
    >>> pipeline = Pipeline([
    ...     ("imputer", Imputer()),
    ...     ("scaler", Scaler()),
    ...     ("encoder", Encoder())
    ... ])
    >>> pipeline.fit(df, numerical_cols=["age"], categorical_cols=["category"])
    >>> result = pipeline.transform(df)
    """

    # Step types for validation
    PREPROCESSING_TYPES: ClassVar[Set[str]] = {"Imputer", "Scaler", "Encoder"}
    MODEL_TYPES: ClassVar[Set[str]] = {
        "BaseModel",
        "BaseRegressor",
        "BaseClassifier",
        "RandomForestRegressor",
        "RandomForestClassifier",
        "XGBoostRegressor",
        "XGBoostClassifier",
        "LightGBMRegressor",
        "LightGBMClassifier",
        "CatBoostRegressor",
        "CatBoostClassifier",
    }

    def __init__(self, steps: List[Tuple[str, Any]]) -> None:
        """Initialize the Pipeline with a list of steps.

        Parameters
        ----------
        steps : List[Tuple[str, Any]]
            A list of (name, step) tuples. Names must be unique strings.

        Raises
        ------
        ValidationError
            If steps is empty, names are not unique, or step order is invalid.
        """
        self._validate_steps(steps)
        self._steps = steps
        self._is_fitted = False
        self._has_model = self._check_has_model()
        self._numerical_cols: Optional[List[str]] = None
        self._categorical_cols: Optional[List[str]] = None
        self._label_col: Optional[str] = None

    def _validate_steps(self, steps: List[Tuple[str, Any]]) -> None:
        """Validate the pipeline steps.

        Parameters
        ----------
        steps : List[Tuple[str, Any]]
            The steps to validate.

        Raises
        ------
        ValidationError
            If steps is empty, names are not unique, or step order is invalid.
        """
        if not steps:
            raise ValidationError(message="Pipeline steps cannot be empty")

        # Check that steps is a list of tuples
        if not isinstance(steps, list):
            raise ValidationError(message="steps must be a list of (name, step) tuples")

        for step in steps:
            if not isinstance(step, tuple) or len(step) != 2:
                raise ValidationError(
                    message="Each step must be a (name, step) tuple"
                )
            name, obj = step
            if not isinstance(name, str):
                raise ValidationError(
                    message=f"Step name must be a string, got {type(name).__name__}"
                )
            if not name:
                raise ValidationError(message="Step name cannot be empty")

        # Check for unique names
        names = [name for name, _ in steps]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValidationError(
                message=f"Step names must be unique. Duplicates found: {set(duplicates)}"
            )

        # Validate step order: preprocessing steps should come before model
        self._validate_step_order(steps)

    def _validate_step_order(self, steps: List[Tuple[str, Any]]) -> None:
        """Validate that preprocessing steps come before model steps.

        Parameters
        ----------
        steps : List[Tuple[str, Any]]
            The steps to validate order.

        Raises
        ------
        ValidationError
            If a preprocessing step appears after a model step.
        """
        model_found = False
        model_name = None

        for name, step in steps:
            step_type = type(step).__name__

            if self._is_model_step(step):
                model_found = True
                model_name = name
            elif self._is_preprocessing_step(step) and model_found:
                raise ValidationError(
                    message=(
                        f"Invalid step order: preprocessing step '{name}' "
                        f"cannot appear after model step '{model_name}'. "
                        "All preprocessing steps must come before model steps."
                    )
                )

    def _is_preprocessing_step(self, step: Any) -> bool:
        """Check if a step is a preprocessing step.

        Parameters
        ----------
        step : Any
            The step to check.

        Returns
        -------
        bool
            True if the step is a preprocessing step.
        """
        step_type = type(step).__name__
        if step_type in self.PREPROCESSING_TYPES:
            return True
        # Also check for duck typing - has fit and transform but not predict
        return (
            hasattr(step, "fit")
            and hasattr(step, "transform")
            and not hasattr(step, "predict")
        )

    def _is_model_step(self, step: Any) -> bool:
        """Check if a step is a model step.

        Parameters
        ----------
        step : Any
            The step to check.

        Returns
        -------
        bool
            True if the step is a model step.
        """
        step_type = type(step).__name__
        if step_type in self.MODEL_TYPES:
            return True
        # Also check for duck typing - has fit and predict
        return hasattr(step, "fit") and hasattr(step, "predict")

    def _check_has_model(self) -> bool:
        """Check if the pipeline contains a model step.

        Returns
        -------
        bool
            True if the pipeline has a model step.
        """
        for _, step in self._steps:
            if self._is_model_step(step):
                return True
        return False

    @property
    def steps(self) -> List[Tuple[str, Any]]:
        """Return the list of pipeline steps.

        Returns
        -------
        List[Tuple[str, Any]]
            Copy of the steps list.
        """
        return list(self._steps)

    @property
    def named_steps(self) -> Dict[str, Any]:
        """Return the pipeline steps as a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary mapping step names to step objects.
        """
        return {name: step for name, step in self._steps}

    def fit(
        self,
        df: DataFrame,
        label_col: Optional[str] = None,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> "Pipeline":
        """Fit all steps in the pipeline.

        Parameters
        ----------
        df : DataFrame
            The PySpark DataFrame to fit on.
        label_col : str, optional
            The label column name, required if pipeline contains a model.
        numerical_cols : List[str], optional
            List of numerical column names for preprocessing steps.
        categorical_cols : List[str], optional
            List of categorical column names for preprocessing steps.

        Returns
        -------
        Pipeline
            Returns self for method chaining.

        Raises
        ------
        ValidationError
            If label_col is missing when pipeline contains a model.
        """
        # Store column configurations
        self._numerical_cols = numerical_cols
        self._categorical_cols = categorical_cols
        self._label_col = label_col

        # Validate label_col if model is present
        if self._has_model and label_col is None:
            raise ValidationError(
                message="label_col is required when pipeline contains a model"
            )

        current_df = df

        for name, step in self._steps:
            if self._is_model_step(step):
                # Model step - use label_col and feature cols
                feature_cols = self._get_feature_cols(current_df, label_col)
                step.fit(current_df, label_col=label_col, feature_cols=feature_cols)
            else:
                # Preprocessing step - pass appropriate columns
                current_df = self._fit_preprocessing_step(
                    step, current_df, numerical_cols, categorical_cols
                )

        self._is_fitted = True
        return self

    def _fit_preprocessing_step(
        self,
        step: Any,
        df: DataFrame,
        numerical_cols: Optional[List[str]],
        categorical_cols: Optional[List[str]],
    ) -> DataFrame:
        """Fit a preprocessing step and return the transformed DataFrame.

        Parameters
        ----------
        step : Any
            The preprocessing step to fit.
        df : DataFrame
            The DataFrame to fit on.
        numerical_cols : List[str], optional
            Numerical columns for the step.
        categorical_cols : List[str], optional
            Categorical columns for the step.

        Returns
        -------
        DataFrame
            The transformed DataFrame (for column tracking).
        """
        step_type = type(step).__name__

        if step_type == "Imputer":
            step.fit(df, numerical_cols=numerical_cols, categorical_cols=categorical_cols)
            return step.transform(df)
        elif step_type == "Scaler":
            if numerical_cols:
                # Only fit if there are numerical columns
                valid_cols = [c for c in numerical_cols if c in df.columns]
                if valid_cols:
                    step.fit(df, numerical_cols=valid_cols)
                    return step.transform(df)
            return df
        elif step_type == "Encoder":
            if categorical_cols:
                # Only fit if there are categorical columns
                valid_cols = [c for c in categorical_cols if c in df.columns]
                if valid_cols:
                    step.fit(df, categorical_cols=valid_cols)
                    return step.transform(df)
            return df
        else:
            # Generic preprocessing step with fit and transform
            # Try various common signatures
            try:
                step.fit(df, numerical_cols=numerical_cols, categorical_cols=categorical_cols)
            except TypeError:
                try:
                    step.fit(df)
                except TypeError:
                    pass
            return step.transform(df) if hasattr(step, "transform") else df

    def _get_feature_cols(
        self, df: DataFrame, label_col: Optional[str]
    ) -> List[str]:
        """Get feature columns (all columns except label).

        Parameters
        ----------
        df : DataFrame
            The DataFrame to get columns from.
        label_col : str, optional
            The label column to exclude.

        Returns
        -------
        List[str]
            List of feature column names.
        """
        cols = df.columns
        if label_col and label_col in cols:
            cols = [c for c in cols if c != label_col]
        return cols

    def transform(self, df: DataFrame) -> DataFrame:
        """Transform data using all preprocessing steps in the pipeline.

        Parameters
        ----------
        df : DataFrame
            The PySpark DataFrame to transform.

        Returns
        -------
        DataFrame
            The transformed DataFrame.

        Raises
        ------
        ModelNotFittedError
            If transform is called before fit.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Pipeline has not been fitted. Call fit() first."
            )

        current_df = df

        for name, step in self._steps:
            if self._is_model_step(step):
                # Skip model steps in transform - use predict instead
                break
            else:
                # Preprocessing step
                current_df = step.transform(current_df)

        return current_df

    def fit_transform(
        self,
        df: DataFrame,
        label_col: Optional[str] = None,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> DataFrame:
        """Fit the pipeline and transform the data in one step.

        Parameters
        ----------
        df : DataFrame
            The PySpark DataFrame to fit and transform.
        label_col : str, optional
            The label column name.
        numerical_cols : List[str], optional
            List of numerical column names.
        categorical_cols : List[str], optional
            List of categorical column names.

        Returns
        -------
        DataFrame
            The transformed DataFrame.
        """
        self.fit(
            df,
            label_col=label_col,
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
        )
        return self.transform(df)

    def predict(self, df: DataFrame) -> DataFrame:
        """Make predictions using the pipeline's model.

        First applies all preprocessing steps, then uses the model to predict.

        Parameters
        ----------
        df : DataFrame
            The PySpark DataFrame to make predictions on.

        Returns
        -------
        DataFrame
            DataFrame with predictions added.

        Raises
        ------
        ModelNotFittedError
            If predict is called before fit.
        ValidationError
            If the pipeline does not contain a model.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Pipeline has not been fitted. Call fit() first."
            )

        if not self._has_model:
            raise ValidationError(
                message="Pipeline does not contain a model. predict() is only available for pipelines with models."
            )

        # Apply all preprocessing steps
        current_df = self.transform(df)

        # Find and apply the model
        for name, step in self._steps:
            if self._is_model_step(step):
                return step.predict(current_df)

        # Should not reach here if _has_model is True
        raise ValidationError(message="No model found in pipeline")

    def save(self, path: str) -> None:
        """Save the pipeline to disk.

        Parameters
        ----------
        path : str
            The directory path to save the pipeline to.

        Raises
        ------
        ModelNotFittedError
            If save is called before fit.
        """
        if not self._is_fitted:
            raise ModelNotFittedError(
                "Pipeline has not been fitted. Call fit() before saving."
            )

        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Save metadata
        metadata = {
            "numerical_cols": self._numerical_cols,
            "categorical_cols": self._categorical_cols,
            "label_col": self._label_col,
            "has_model": self._has_model,
            "step_names": [name for name, _ in self._steps],
            "step_types": [type(step).__name__ for _, step in self._steps],
        }

        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # Save each step
        for i, (name, step) in enumerate(self._steps):
            step_path = os.path.join(path, f"step_{i}_{name}")

            if hasattr(step, "save"):
                # Step has its own save method
                step.save(step_path)
            else:
                # Pickle the step
                with open(step_path + ".pkl", "wb") as f:
                    pickle.dump(step, f)

    @classmethod
    def load(cls, path: str) -> "Pipeline":
        """Load a pipeline from disk.

        Parameters
        ----------
        path : str
            The directory path to load the pipeline from.

        Returns
        -------
        Pipeline
            The loaded pipeline.
        """
        # Load metadata
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)

        step_names = metadata["step_names"]
        step_types = metadata["step_types"]

        # Load each step
        steps = []
        for i, (name, step_type) in enumerate(zip(step_names, step_types)):
            step_path = os.path.join(path, f"step_{i}_{name}")

            # Try loading with step's load method first
            step = cls._load_step(step_path, step_type)
            steps.append((name, step))

        # Create pipeline without validation (steps are already validated)
        pipeline = object.__new__(cls)
        pipeline._steps = steps
        pipeline._is_fitted = True
        pipeline._has_model = metadata["has_model"]
        pipeline._numerical_cols = metadata["numerical_cols"]
        pipeline._categorical_cols = metadata["categorical_cols"]
        pipeline._label_col = metadata["label_col"]

        return pipeline

    @classmethod
    def _load_step(cls, step_path: str, step_type: str) -> Any:
        """Load a single step from disk.

        Parameters
        ----------
        step_path : str
            The path to the step.
        step_type : str
            The type name of the step.

        Returns
        -------
        Any
            The loaded step object.
        """
        # Try pickle first (most common case)
        pickle_path = step_path + ".pkl"
        if os.path.exists(pickle_path):
            with open(pickle_path, "rb") as f:
                return pickle.load(f)

        # Try known step types with their own load methods
        if step_type == "Imputer":
            from smallaxe.preprocessing import Imputer

            return Imputer.load(step_path)
        elif step_type == "Scaler":
            from smallaxe.preprocessing import Scaler

            return Scaler.load(step_path)
        elif step_type == "Encoder":
            from smallaxe.preprocessing import Encoder

            return Encoder.load(step_path)

        # Fallback - try directory-based load
        if os.path.isdir(step_path):
            # Check if there's a pickle file inside
            for filename in os.listdir(step_path):
                if filename.endswith(".pkl"):
                    with open(os.path.join(step_path, filename), "rb") as f:
                        return pickle.load(f)

        raise ValueError(f"Could not load step at {step_path}")

    def __repr__(self) -> str:
        """Return a string representation of the pipeline."""
        step_strs = [f"('{name}', {type(step).__name__})" for name, step in self._steps]
        steps_repr = ", ".join(step_strs)
        fitted_str = "fitted" if self._is_fitted else "not fitted"
        return f"Pipeline([{steps_repr}], {fitted_str})"

    def __len__(self) -> int:
        """Return the number of steps in the pipeline."""
        return len(self._steps)

    def __getitem__(self, key: str) -> Any:
        """Get a step by name.

        Parameters
        ----------
        key : str
            The step name.

        Returns
        -------
        Any
            The step object.

        Raises
        ------
        KeyError
            If the step name is not found.
        """
        for name, step in self._steps:
            if name == key:
                return step
        raise KeyError(f"Step '{key}' not found in pipeline")

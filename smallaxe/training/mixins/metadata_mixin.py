"""MetadataMixin for capturing and storing training metadata."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pyspark.sql import DataFrame

import smallaxe
from smallaxe.exceptions import ModelNotFittedError


class MetadataMixin:
    """Mixin providing metadata capture and storage functionality.

    This mixin captures and stores metadata about the training process,
    including timestamps, Spark version, column information, and data statistics.
    """

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get training metadata.

        Returns:
            Dictionary containing training metadata:
            - training_timestamp: When the model was trained
            - spark_version: Version of Spark used for training
            - smallaxe_version: Version of smallaxe used
            - label_col: Name of the label column
            - feature_cols: List of feature column names
            - n_samples: Number of training samples
            - n_features: Number of features
            - seed: Random seed used (if set)

        Raises:
            ModelNotFittedError: If accessed before the model is fitted.
        """
        if not getattr(self, "_metadata", None):
            raise ModelNotFittedError(
                "No metadata available. Model has not been fitted."
            )
        return self._metadata.copy()

    def _capture_metadata(
        self,
        df: DataFrame,
        label_col: str,
        feature_cols: List[str],
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Capture metadata from the training process.

        Args:
            df: Training DataFrame.
            label_col: Name of the label column.
            feature_cols: List of feature column names.
            extra_metadata: Additional metadata to include.
        """
        # Get Spark version
        spark_version = None
        try:
            spark = df.sparkSession
            spark_version = spark.version
        except Exception:
            pass

        # Capture metadata
        self._metadata: Dict[str, Any] = {
            "training_timestamp": datetime.now().isoformat(),
            "spark_version": spark_version,
            "smallaxe_version": smallaxe.__version__,
            "label_col": label_col,
            "feature_cols": feature_cols,
            "n_samples": df.count(),
            "n_features": len(feature_cols),
            "seed": smallaxe.get_seed(),
        }

        # Add extra metadata if provided
        if extra_metadata:
            self._metadata.update(extra_metadata)

    def _capture_label_stats(
        self,
        df: DataFrame,
        label_col: str,
        task_type: str,
    ) -> Dict[str, Any]:
        """Capture statistics about the label column.

        Args:
            df: DataFrame containing the label column.
            label_col: Name of the label column.
            task_type: Type of task ('regression', 'binary', 'multiclass').

        Returns:
            Dictionary with label statistics.
        """
        from pyspark.sql import functions as F

        stats: Dict[str, Any] = {}

        if task_type == "regression":
            # Regression stats
            label_stats = df.select(
                F.min(label_col).alias("min"),
                F.max(label_col).alias("max"),
                F.mean(label_col).alias("mean"),
                F.stddev(label_col).alias("stddev"),
            ).first()

            if label_stats:
                stats["label_min"] = float(label_stats["min"]) if label_stats["min"] is not None else None
                stats["label_max"] = float(label_stats["max"]) if label_stats["max"] is not None else None
                stats["label_mean"] = float(label_stats["mean"]) if label_stats["mean"] is not None else None
                stats["label_stddev"] = float(label_stats["stddev"]) if label_stats["stddev"] is not None else None

        else:
            # Classification stats
            class_counts = (
                df.groupBy(label_col)
                .count()
                .collect()
            )
            stats["class_counts"] = {
                str(row[label_col]): row["count"] for row in class_counts
            }
            stats["n_classes"] = len(class_counts)

        return stats

    def _update_metadata(self, key: str, value: Any) -> None:
        """Update a specific metadata field.

        Args:
            key: Metadata field name.
            value: Value to set.
        """
        if not hasattr(self, "_metadata"):
            self._metadata = {}
        self._metadata[key] = value

    def _get_metadata_for_persistence(self) -> Dict[str, Any]:
        """Get metadata in a format suitable for persistence.

        Returns:
            Dictionary of metadata suitable for JSON serialization.
        """
        if not hasattr(self, "_metadata"):
            return {}
        return self._metadata.copy()

    def _restore_metadata_from_persistence(self, metadata: Dict[str, Any]) -> None:
        """Restore metadata from persisted state.

        Args:
            metadata: Dictionary of metadata loaded from persistence.
        """
        self._metadata = metadata.copy()

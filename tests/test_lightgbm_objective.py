"""LightGBM objective selection, tested without SynapseML installed."""

import pytest

from smallaxe.training import lightgbm as lightgbm_module


class _FakeSparkLightGBMClassifier:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def setSeed(self, seed):
        self.kwargs["seed"] = seed


@pytest.mark.parametrize("task, objective", [("binary", "binary"), ("multiclass", "multiclass")])
def test_classifier_objective_matches_task(monkeypatch, task, objective):
    """SynapseML defaults to 'binary', so multiclass must be requested explicitly."""
    monkeypatch.setattr(lightgbm_module, "LIGHTGBM_AVAILABLE", True)
    monkeypatch.setattr(lightgbm_module, "SparkLightGBMClassifier", _FakeSparkLightGBMClassifier)

    model = lightgbm_module.LightGBMClassifier(task=task)
    estimator = model._create_spark_estimator(
        features_col="features", label_col="label", prediction_col="prediction"
    )

    assert estimator.kwargs["objective"] == objective

"""Model charts: feature importances and a comparison of several models on one metric.

Neither touches a DataFrame: the importances come from ``Model.feature_importances``,
and the comparison from ``AutomatedTraining.comparison`` or a mapping of scores. The
metric orientation (which direction is better) is the one ``optimize.run`` and
``AutomatedTraining`` use, defined in ``smallaxe.search.optimize``.
"""

import math
import numbers
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Tuple, Union

from smallaxe.exceptions import ValidationError
from smallaxe.viz import _style

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

    from smallaxe.auto import AutomatedTraining
    from smallaxe.pipeline import Pipeline
    from smallaxe.training import Model

__all__ = ["feature_importances", "model_comparison"]


# --- Feature importances --------------------------------------------------------------


def _importances(model: Any) -> Tuple[Dict[str, float], str]:
    """The importances of a fitted model, or of a fitted Pipeline's model step.

    Returns:
        ``(feature name -> importance, algorithm name)``.

    Raises:
        ValidationError: If ``model`` is neither a smallaxe Model nor a Pipeline, the
            Pipeline has no smallaxe model step, or the algorithm reports none.
        ModelNotFittedError: If the model is not fitted.
    """
    from smallaxe.pipeline import Pipeline
    from smallaxe.training import Model

    if isinstance(model, Pipeline):
        steps = [step for _, step in model.steps if isinstance(step, Model)]
        if not steps:
            raise ValidationError(
                "feature_importances needs a Pipeline with a smallaxe model step; this "
                "Pipeline has none."
            )
        model = steps[-1]
    elif not isinstance(model, Model):
        raise ValidationError(
            "feature_importances needs a fitted smallaxe Model or Pipeline, got "
            f"{type(model).__name__}."
        )
    importances = model.feature_importances  # ModelNotFittedError before fit
    name = model.algorithm.name
    if importances is None:
        raise ValidationError(f"The {name!r} model reports no feature importances.")
    return {str(key): float(value) for key, value in importances.items()}, name


def feature_importances(
    model: Union["Model", "Pipeline"], *, top_n: int = 20, title: Optional[str] = None
) -> "Figure":
    """Horizontal bars of a fitted model's feature importances, largest at the top.

    The values are ``model.feature_importances``, as the algorithm defines them (Random
    Forest: normalised Gini importance; XGBoost: total gain; LightGBM: split counts;
    CatBoost: PredictionValuesChange). Only the ``top_n`` largest are shown; equal
    importances are ordered by feature name.

    Args:
        model: A fitted smallaxe model (from ``regressors`` / ``classifiers``, a
            ``best_model``, or ``load``), or a fitted Pipeline whose model step is one.
        top_n: How many features to show, the largest first: a positive integer.
            Default 20.
        title: The chart title. Defaults to "Feature importances (<algorithm>)".

    Returns:
        A ``plotly.graph_objects.Figure`` holding one horizontal bar trace: ``y`` the
        feature names and ``x`` their importances, largest first.

    Raises:
        DependencyError: If plotly is not installed (``pip install smallaxe[viz]``).
        ModelNotFittedError: If the model (or the Pipeline's model step) is not fitted.
        ValidationError: If ``model`` is neither a smallaxe Model nor a Pipeline with a
            model step, its algorithm reports no importances, or ``top_n`` or ``title``
            is invalid.

    Example:
        >>> from smallaxe import viz
        >>> viz.feature_importances(model, top_n=10).show()
    """
    go = _style.plotly()
    _style.check_title(title)
    top_n = _style.check_int("top_n", top_n, 1)
    importances, algorithm = _importances(model)
    ranked = sorted(importances.items(), key=lambda item: (-item[1], item[0]))
    shown = ranked[:top_n]
    subtitle = f"top {len(shown)} of {len(ranked)} features" if len(shown) < len(ranked) else None

    fig = _style.figure(
        go,
        title=title,
        default_title=f"Feature importances ({algorithm})",
        x_title="Importance",
        y_title="Feature",
        subtitle=subtitle,
    )
    fig.add_trace(
        go.Bar(
            x=[value for _, value in shown],
            y=[_style.text(name) for name, _ in shown],
            orientation="h",
            marker={"color": _style.ACCENT},
            text=[_style.number(value) for _, value in shown],
            textposition="auto",
            name="Importance",
            hovertemplate="%{y}<br>Importance %{x:.4g}<extra></extra>",
        )
    )
    # The first bar (the largest) at the top.
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_layout(showlegend=False)
    return fig


# --- Model comparison -----------------------------------------------------------------


def _orientation(metric: str) -> bool:
    """Whether a higher ``metric`` is better, by the library's one orientation rule.

    A k-fold score key (``mean_rmse``) has its metric's orientation.

    Raises:
        ValidationError: If ``metric`` is not a metric smallaxe knows the direction of.
    """
    from smallaxe.search.optimize import _VALID_METRICS, _higher_is_better

    base = metric
    if base not in _VALID_METRICS and base.startswith("mean_"):
        base = base[len("mean_") :]
    if base not in _VALID_METRICS:
        raise ValidationError(
            f"Cannot compare on {metric!r}: it is not a metric smallaxe knows the direction "
            f"of. Use one of {sorted(_VALID_METRICS)}, or its k-fold 'mean_' form."
        )
    return _higher_is_better(base)


def _score(value: Any, name: str, metric: str) -> Optional[float]:
    """A score as a float, or None when it is undefined (None or NaN).

    Raises:
        ValidationError: If ``value`` is not a number.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValidationError(
            f"The {metric!r} score of {name!r} must be a number or None, got {value!r}."
        )
    score = float(value)
    return None if math.isnan(score) else score


def _comparison_scores(results: Any, metric: Any) -> Tuple[Dict[str, Optional[float]], str]:
    """``(model name -> score or None, metric)`` from an AutomatedTraining or a mapping.

    Raises:
        ModelNotFittedError: If ``results`` is an AutomatedTraining that is not fitted.
        ValidationError: If ``results`` is neither, ``metric`` is invalid or missing
            from every entry, or the mapping is empty or malformed.
    """
    from smallaxe.auto import AutomatedTraining
    from smallaxe.metrics import TASK_METRICS

    if metric is not None and not isinstance(metric, str):
        raise ValidationError(f"metric must be a metric name or None, got {metric!r}.")

    if isinstance(results, AutomatedTraining):
        table = results.comparison  # ModelNotFittedError before fit
        metric = results.metric if metric is None else metric
        if metric not in TASK_METRICS[results.task]:
            raise ValidationError(
                f"The comparison of this AutomatedTraining has no metric {metric!r}. Its "
                f"metrics are: {sorted(TASK_METRICS[results.task])}."
            )
        scores = {
            str(row["algorithm"]): _score(row[metric], str(row["algorithm"]), metric)
            for row in table.to_dict("records")
            if row["status"] == "ok"
        }
        return scores, metric

    if not isinstance(results, Mapping):
        raise ValidationError(
            "model_comparison needs a fitted AutomatedTraining or a mapping of model name "
            f"to a mapping of metric name to score, got {type(results).__name__}."
        )
    if metric is None:
        raise ValidationError(
            "model_comparison needs metric= when results is a mapping, for example "
            "metric='rmse'."
        )
    if not results:
        raise ValidationError("model_comparison needs at least one model; results is empty.")
    for name, entry in results.items():
        if not isinstance(entry, Mapping):
            raise ValidationError(
                f"results[{name!r}] must be a mapping of metric name to score, got "
                f"{type(entry).__name__}."
            )
    if not any(metric in entry for entry in results.values()):
        raise ValidationError(f"No entry of results has a {metric!r} score.")
    scores = {
        str(name): _score(entry.get(metric), str(name), metric) for name, entry in results.items()
    }
    return scores, metric


def model_comparison(
    results: Union["AutomatedTraining", Mapping[str, Mapping[str, Any]]],
    metric: Optional[str] = None,
    *,
    title: Optional[str] = None,
) -> "Figure":
    """One bar per model on one metric, best first, the best bar highlighted.

    "Best" follows the metric's orientation, the one ``optimize.run`` and
    ``AutomatedTraining`` use: ``rmse``, ``mse``, ``mae``, ``mape`` and ``log_loss`` are
    minimised, the others maximised. A model whose score is undefined (None or NaN) is
    left out and named in the subtitle.

    Args:
        results: A fitted :class:`~smallaxe.auto.AutomatedTraining` (its ``comparison``
            rows with status ``ok``), or a mapping of model name to a mapping of metric
            name to score, such as ``{"rf": rf.validation_scores, "xgb":
            xgb.validation_scores}``.
        metric: The metric to compare on: a key of the scores (a k-fold
            ``validation_scores`` key such as ``"mean_rmse"`` has its metric's
            orientation). Defaults to the AutomatedTraining's metric; required for a
            mapping.
        title: The chart title. Defaults to "Model comparison on <metric>".

    Returns:
        A ``plotly.graph_objects.Figure`` holding one bar trace: ``x`` the model names,
        best first, and ``y`` their scores.

    Raises:
        DependencyError: If plotly is not installed (``pip install smallaxe[viz]``).
        ModelNotFittedError: If ``results`` is an AutomatedTraining that is not fitted.
        ValidationError: If ``results`` is neither a fitted AutomatedTraining nor a
            non-empty mapping of mappings, ``metric`` is missing for a mapping, is not a
            metric smallaxe knows the direction of, or is in no entry, a score is not a
            number, every score is undefined, or ``title`` is invalid.

    Example:
        >>> from smallaxe import viz
        >>> viz.model_comparison(auto).show()
        >>> viz.model_comparison({"rf": rf.validation_scores, "xgb": xgb.validation_scores},
        ...                      metric="rmse")
    """
    go = _style.plotly()
    _style.check_title(title)
    scores, metric = _comparison_scores(results, metric)
    maximize = _orientation(metric)
    defined = [(name, score) for name, score in scores.items() if score is not None]
    undefined = [name for name, score in scores.items() if score is None]
    if not defined:
        raise ValidationError(f"Every {metric!r} score is undefined: {undefined}.")
    ranked = sorted(defined, key=lambda item: -item[1] if maximize else item[1])

    direction = "higher is better" if maximize else "lower is better"
    subtitle = direction
    if undefined:
        subtitle += "; left out (score undefined): " + ", ".join(
            _style.text(name) for name in undefined
        )
    fig = _style.figure(
        go,
        title=title,
        default_title=f"Model comparison on {_style.text(metric)}",
        x_title="Model",
        y_title=_style.text(metric),
        subtitle=subtitle,
    )
    fig.add_trace(
        go.Bar(
            x=[_style.text(name) for name, _ in ranked],
            y=[score for _, score in ranked],
            marker={
                "color": [
                    _style.ACCENT if index == 0 else _style.NEUTRAL for index in range(len(ranked))
                ]
            },
            text=[_style.number(score) for _, score in ranked],
            textposition="outside",
            name=metric,
            hovertemplate="%{x}<br>" + _style.text(metric) + " %{y:.4g}<extra></extra>",
        )
    )
    fig.update_xaxes(type="category")
    fig.update_layout(showlegend=False)
    return fig

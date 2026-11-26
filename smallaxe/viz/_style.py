"""What every chart shares: the lazy plotly import, the layout, and the argument checks.

plotly is optional (``smallaxe[viz]``), so it is imported here, inside :func:`plotly`,
and nowhere else: ``import smallaxe.viz`` never imports it, and a chart called without
it raises ``DependencyError`` before any Spark job runs.

The appearance is plotly's built-in ``plotly_white`` template, one sequential colorscale
for heatmaps, one accent colour for the data and a neutral grey for reference lines and
the bars that are not highlighted. Nothing is loaded from the network.
"""

import html
from typing import TYPE_CHECKING, Any, Optional

from smallaxe.exceptions import DependencyError, ValidationError

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

__all__ = [
    "ACCENT",
    "COLORSCALE",
    "NEUTRAL",
    "TEMPLATE",
    "plotly",
    "check_title",
    "check_int",
    "check_bool",
    "text",
    "number",
    "figure",
]

_INSTALL_COMMAND = "pip install smallaxe[viz]"

TEMPLATE = "plotly_white"
# The one sequential colorscale (heatmaps) and the one accent colour (curves, bars).
COLORSCALE = "Blues"
ACCENT = "#1f77b4"
# Reference lines (chance, baseline, y = x, zero) and bars that are not highlighted.
NEUTRAL = "#8c96a0"


def plotly() -> Any:
    """``plotly.graph_objects``, imported on first use.

    Returns:
        The ``plotly.graph_objects`` module.

    Raises:
        DependencyError: If plotly is not installed or cannot be imported.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise DependencyError(package="plotly", install_command=_INSTALL_COMMAND) from exc
    return go


def check_title(title: Any) -> Optional[str]:
    """``title`` if it is None or a string.

    Raises:
        ValidationError: Otherwise.
    """
    if title is not None and not isinstance(title, str):
        raise ValidationError(f"title must be a string or None, got {type(title).__name__}.")
    return title


def check_int(name: str, value: Any, low: int, high: Optional[int] = None) -> int:
    """``value`` if it is an int (not a bool) in ``[low, high]``.

    Raises:
        ValidationError: Otherwise.
    """
    in_range = isinstance(value, int) and not isinstance(value, bool) and value >= low
    if in_range and high is not None:
        in_range = value <= high
    if not in_range:
        bounds = f"between {low} and {high}" if high is not None else f"of at least {low}"
        raise ValidationError(f"{name} must be an integer {bounds}, got {value!r}.")
    return int(value)


def check_bool(name: str, value: Any) -> bool:
    """``value`` if it is a bool.

    Raises:
        ValidationError: Otherwise.
    """
    if not isinstance(value, bool):
        raise ValidationError(f"{name} must be True or False, got {value!r}.")
    return value


def text(value: str) -> str:
    """``value`` escaped for plotly's HTML-like labels (a column name holding ``<``)."""
    return html.escape(value, quote=False)


def number(value: float) -> str:
    """A score or statistic for a label: four significant digits."""
    return f"{value:.4g}"


def figure(
    go: Any,
    *,
    title: Optional[str],
    default_title: str,
    x_title: str,
    y_title: str,
    subtitle: Optional[str] = None,
) -> "Figure":
    """A new figure with the shared layout.

    Args:
        go: ``plotly.graph_objects``, from :func:`plotly`.
        title: The caller's ``title=``; it replaces ``default_title`` when given.
        default_title: The chart's own title.
        x_title: The x axis title.
        y_title: The y axis title.
        subtitle: A line of detail under the title (statistics, notes), or None.

    Returns:
        An empty ``plotly.graph_objects.Figure`` with the layout applied.
    """
    heading = default_title if title is None else title
    if subtitle:
        heading += f"<br><sup>{subtitle}</sup>"
    fig = go.Figure()
    fig.update_layout(
        template=TEMPLATE,
        title={"text": heading, "x": 0.0, "xanchor": "left"},
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend={"bgcolor": "rgba(255,255,255,0.6)"},
    )
    return fig

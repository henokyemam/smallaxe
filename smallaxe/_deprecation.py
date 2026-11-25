"""Deprecation helpers for the spellings smallaxe 1.0 removes.

Every deprecation goes through :func:`warn`: a ``DeprecationWarning`` whose message
names the replacement and says the old spelling is removed in smallaxe 1.0,
attributed to the first caller outside smallaxe (see :func:`caller_stacklevel`).

- :func:`df_keyword` lets a callable whose DataFrame parameter is now called
  ``dataframe`` still accept ``df=``.
- :func:`keyword_only` lets a callable whose trailing parameters became keyword-only
  still accept them positionally.
- :func:`module_getattr` / :func:`module_dir` keep a module's renamed or private
  names importable (PEP 562).

In smallaxe 1.0 the decorators are deleted together with what they keep working.
"""

import functools
import inspect
import sys
import warnings
from types import FrameType
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, TypeVar, cast

from smallaxe.exceptions import ValidationError

_Callable = TypeVar("_Callable")

REMOVED_IN = "smallaxe 1.0"


def _is_smallaxe_frame(frame: FrameType) -> bool:
    """Whether ``frame`` runs code of the smallaxe package itself."""
    name = frame.f_globals.get("__name__", "")
    return isinstance(name, str) and (name == "smallaxe" or name.startswith("smallaxe."))


def caller_stacklevel() -> int:
    """The ``stacklevel`` that attributes a warning to the first frame outside smallaxe.

    Meant for the function that calls ``warnings.warn``: level 1 is that function, so
    the count starts at its caller and skips every smallaxe frame (wrappers, shims,
    nested library calls). smallaxe's own warnings use it, so a warning points at the
    user's line however deep inside smallaxe it was raised.

    Returns:
        The stacklevel to pass to ``warnings.warn``.
    """
    frame = sys._getframe(2)  # 0: this function, 1: the warning function, 2: its caller
    level = 2
    while frame.f_back is not None and _is_smallaxe_frame(frame):
        frame = frame.f_back
        level += 1
    return level


def warn(message: str) -> None:
    """Emit one ``DeprecationWarning`` attributed to the first caller outside smallaxe.

    Args:
        message: The full message; it names the replacement and says the old spelling
            is removed in smallaxe 1.0.
    """
    warnings.warn(message, DeprecationWarning, stacklevel=caller_stacklevel())


def _wrap_descriptor(
    func: Any, decorate: Callable[[Callable[..., Any]], Callable[..., Any]]
) -> Any:
    """Apply ``decorate`` to a function, or to the function inside a class/staticmethod."""
    if isinstance(func, (classmethod, staticmethod)):
        return type(func)(decorate(func.__func__))
    return decorate(func)


def df_keyword(func: _Callable) -> _Callable:
    """Accept the deprecated ``df=`` keyword for a callable's ``dataframe`` parameter.

    The wrapper keeps the callable's name, docstring and ``inspect.signature`` (which
    shows ``dataframe``, not ``df``). Positional calls and ``dataframe=`` pass straight
    through. ``df=`` is renamed to ``dataframe=`` with one ``DeprecationWarning``;
    ``df=`` together with a ``dataframe`` argument raises ``ValidationError``; with
    neither, the callable's own missing-argument ``TypeError`` is raised. Works on
    functions, methods, classmethods and staticmethods.

    Args:
        func: The callable, which must have a parameter named ``dataframe``.

    Returns:
        The wrapped callable.
    """

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        names = list(inspect.signature(function).parameters)
        position = names.index("dataframe")
        label = function.__qualname__

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if "df" in kwargs:
                if "dataframe" in kwargs or len(args) > position:
                    raise ValidationError(
                        f"{label}() got both df= and dataframe; pass only dataframe "
                        "(df= is deprecated)."
                    )
                warn(
                    f"The df= keyword of {label}() is deprecated and will be removed in "
                    f"{REMOVED_IN}. Use dataframe= or pass the DataFrame positionally."
                )
                kwargs["dataframe"] = kwargs.pop("df")
            return function(*args, **kwargs)

        return wrapper

    return cast(_Callable, _wrap_descriptor(func, decorate))


def keyword_only(*names: str) -> Callable[[_Callable], _Callable]:
    """Accept, with a warning, positional values for parameters that became keyword-only.

    The decorated callable declares ``names`` after ``*``. A call passing them
    positionally, after the positional parameters, still works and emits one
    ``DeprecationWarning``. The wrapper keeps the callable's ``inspect.signature``.

    Args:
        *names: The keyword-only parameters that 0.x accepted positionally, in their
            old positional order.

    Returns:
        A decorator.
    """

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        parameters = inspect.signature(function).parameters.values()
        n_positional = sum(
            1
            for p in parameters
            if p.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        )
        label = function.__qualname__

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            extra = args[n_positional:]
            if extra:
                if len(extra) > len(names):
                    raise TypeError(
                        f"{label}() takes at most {n_positional + len(names)} positional "
                        f"arguments ({len(args)} given)"
                    )
                moved = names[: len(extra)]
                for name, value in zip(moved, extra):
                    if name in kwargs:
                        raise TypeError(f"{label}() got multiple values for argument {name!r}")
                    kwargs[name] = value
                listed = ", ".join(f"{name}=" for name in moved)
                warn(
                    f"Passing {', '.join(moved)} to {label}() positionally is deprecated and "
                    f"will be removed in {REMOVED_IN}. Pass it by keyword ({listed})."
                )
                args = args[:n_positional]
            return function(*args, **kwargs)

        return wrapper

    def apply(func: _Callable) -> _Callable:
        return cast(_Callable, _wrap_descriptor(func, decorate))

    return apply


# A deprecated module attribute: the replacement to name, and how to get its value.
Moved = Tuple[str, Callable[[], Any]]


def _called_by_import_machinery() -> bool:
    """Whether the module ``__getattr__`` two frames up was called by importlib.

    ``from package import name`` makes importlib's ``_handle_fromlist`` probe the name
    with ``hasattr`` before the importing frame reads it; only that read should warn.
    """
    caller = sys._getframe(2)
    return caller.f_globals.get("__name__") == "importlib._bootstrap"


def module_getattr(module: str, moved: Mapping[str, Moved]) -> Callable[[str], Any]:
    """A PEP 562 ``__getattr__`` serving a module's deprecated names.

    Reading a deprecated name emits one ``DeprecationWarning`` naming its replacement and
    returns the value its getter gives; any other unknown name raises ``AttributeError``.

    Args:
        module: The module's ``__name__``, for messages.
        moved: Deprecated name -> ``(replacement, getter)``.

    Returns:
        The function to assign to the module's ``__getattr__``.
    """

    def __getattr__(name: str) -> Any:
        if name not in moved:
            raise AttributeError(f"module {module!r} has no attribute {name!r}")
        replacement, getter = moved[name]
        if not _called_by_import_machinery():
            warn(
                f"{module}.{name} is deprecated and will be removed in {REMOVED_IN}. "
                f"Use {replacement} instead."
            )
        return getter()

    return __getattr__


def module_dir(namespace: Dict[str, Any], moved: Iterable[str]) -> Callable[[], List[str]]:
    """A PEP 562 ``__dir__`` listing a module's names plus its deprecated ones.

    Args:
        namespace: The module's ``globals()``.
        moved: The deprecated names its ``__getattr__`` serves.

    Returns:
        The function to assign to the module's ``__dir__``.
    """
    extra = list(moved)

    def __dir__() -> List[str]:
        return sorted(set(namespace) | set(extra))

    return __dir__


def renamed(value: Any) -> Callable[[], Any]:
    """A getter for :func:`module_getattr` that returns ``value``."""
    return lambda: value


def with_df_keyword(func: Callable[..., Any]) -> Callable[[], Any]:
    """A getter returning ``func`` wrapped by :func:`df_keyword`, built once.

    For a public helper that became private: its old name still accepts ``df=``.
    """
    wrapped: Optional[Callable[..., Any]] = None

    def getter() -> Any:
        nonlocal wrapped
        if wrapped is None:
            wrapped = df_keyword(func)
        return wrapped

    return getter

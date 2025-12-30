"""ParamMixin for managing model hyperparameters."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from smallaxe.exceptions import ValidationError


class ParamMixin(ABC):
    """Mixin providing hyperparameter management functionality.

    This mixin provides a consistent interface for getting, setting,
    and validating model hyperparameters across all model classes.
    """

    @property
    @abstractmethod
    def params(self) -> Dict[str, str]:
        """Get parameter names and their descriptions.

        Returns:
            Dictionary mapping parameter names to their descriptions.

        Example:
            >>> model.params
            {'n_estimators': 'Number of trees in the forest',
             'max_depth': 'Maximum depth of the tree'}
        """
        pass

    @property
    @abstractmethod
    def default_params(self) -> Dict[str, Any]:
        """Get default parameter values.

        Returns:
            Dictionary mapping parameter names to their default values.

        Example:
            >>> model.default_params
            {'n_estimators': 100, 'max_depth': 5}
        """
        pass

    def set_param(self, params: Dict[str, Any]) -> "ParamMixin":
        """Set one or more model parameters.

        Args:
            params: Dictionary of parameter names and values to set.

        Returns:
            self: The model instance for method chaining.

        Raises:
            ValidationError: If a parameter name is invalid or value fails validation.

        Example:
            >>> model.set_param({'n_estimators': 200, 'max_depth': 10})
        """
        valid_params = set(self.params.keys())

        for name, value in params.items():
            if name not in valid_params:
                raise ValidationError(
                    f"Invalid parameter '{name}'. Valid parameters are: {sorted(valid_params)}"
                )
            self._validate_param(name, value)
            self._set_param_value(name, value)

        return self

    def get_param(self, name: str) -> Any:
        """Get the current value of a parameter.

        Args:
            name: The parameter name.

        Returns:
            The current value of the parameter.

        Raises:
            ValidationError: If the parameter name is invalid.
        """
        valid_params = set(self.params.keys())
        if name not in valid_params:
            raise ValidationError(
                f"Invalid parameter '{name}'. Valid parameters are: {sorted(valid_params)}"
            )
        return self._get_param_value(name)

    def get_params(self) -> Dict[str, Any]:
        """Get all current parameter values.

        Returns:
            Dictionary mapping parameter names to their current values.
        """
        return {name: self._get_param_value(name) for name in self.params.keys()}

    def _validate_param(self, name: str, value: Any) -> None:
        """Validate a parameter value.

        Override this method in subclasses to add custom validation logic.

        Args:
            name: The parameter name.
            value: The value to validate.

        Raises:
            ValidationError: If the value is invalid for the parameter.
        """
        # Default validation: check type matches default
        default_value = self.default_params.get(name)
        if default_value is not None and value is not None:
            if not isinstance(value, type(default_value)):
                # Allow int for float parameters
                if isinstance(default_value, float) and isinstance(value, int):
                    return
                raise ValidationError(
                    f"Parameter '{name}' must be of type {type(default_value).__name__}, "
                    f"got {type(value).__name__}."
                )

    def _set_param_value(self, name: str, value: Any) -> None:
        """Set a single parameter value.

        Default implementation stores parameters in _params dict.
        Override in subclasses if needed.

        Args:
            name: The parameter name.
            value: The value to set.
        """
        if not hasattr(self, "_params"):
            self._params: Dict[str, Any] = {}
        self._params[name] = value

    def _get_param_value(self, name: str) -> Any:
        """Get a single parameter value.

        Default implementation retrieves from _params dict or default_params.
        Override in subclasses if needed.

        Args:
            name: The parameter name.

        Returns:
            The current value of the parameter.
        """
        if hasattr(self, "_params") and name in self._params:
            return self._params[name]
        return self.default_params.get(name)

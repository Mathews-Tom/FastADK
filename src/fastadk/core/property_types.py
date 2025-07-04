"""
Property types for FastADK.

This module provides a collection of validated property types that can be used
for type-safe property management within agents and tools.
"""

import re
from collections.abc import Callable
from typing import Generic, TypeVar

from fastadk.core.exceptions import ValidationError

T = TypeVar("T")


class ValidatedProperty(Generic[T]):
    """Base class for properties with validation.

    This generic class provides a foundation for type-safe property validation
    and can be extended to create specific property types with custom validation
    logic.

    Attributes:
        value: The value of the property.
        validators: A list of validator functions that check if the value is valid.
    """

    def __init__(
        self,
        value: T,
        validators: list[Callable[[T], bool]] | None = None,
        error_message: str = "Validation failed",
    ) -> None:
        """Initialize a validated property.

        Args:
            value: The property value.
            validators: Optional list of validator functions.
            error_message: Message to use when validation fails.
        """
        self.value = value
        self._validators = validators or []
        self._error_message = error_message
        self.validate()

    def validate(self) -> bool:
        """Run all validators against the value.

        Returns:
            True if all validators pass, raises ValueError otherwise.

        Raises:
            ValueError: If any validator fails.
        """
        # Validate by checking each validator function
        for validator in self._validators:
            # Call the validator and explicitly convert to bool
            # to handle any validator type
            if not bool(validator(self.value)):
                raise ValidationError(
                    message=f"{self._error_message}: {self.value}",
                    error_code="PROPERTY_VALIDATION_FAILED",
                    details={
                        "property_type": self.__class__.__name__,
                        "value": str(self.value),
                    },
                )
        # All validators passed
        return True

    def __repr__(self) -> str:
        """Return a string representation of the property."""
        return f"{self.__class__.__name__}({repr(self.value)})"

    def __str__(self) -> str:
        """Return the string representation of the value."""
        return str(self.value)

    def __eq__(self, other: object) -> bool:
        """Compare equality with another object."""
        if isinstance(other, ValidatedProperty):
            return bool(self.value == other.value)
        return bool(self.value == other)


class URLProperty(ValidatedProperty[str]):
    """A URL property with validation.

    This property ensures that the value is a valid URL.
    """

    def __init__(self, value: str) -> None:
        """Initialize a URL property with validation.

        Args:
            value: The URL string to validate.

        Raises:
            ValueError: If the URL is not valid.
        """
        # URL validation regex pattern
        url_pattern = re.compile(
            r"^(https?|ftp)://"  # protocol
            r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|"
            r"[A-Z0-9-]{2,}\.?)|"  # domain
            r"localhost|"  # localhost
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # or IP
            r"(?::\d+)?"  # optional port
            r"(?:/?|[/?]\S+)$",
            re.IGNORECASE,  # path
        )
        validators = [lambda x: bool(url_pattern.match(x))]
        super().__init__(value, validators, "Invalid URL format")


class EmailProperty(ValidatedProperty[str]):
    """An email property with validation.

    This property ensures that the value is a valid email address.
    """

    def __init__(self, value: str) -> None:
        """Initialize an email property with validation.

        Args:
            value: The email string to validate.

        Raises:
            ValueError: If the email is not valid.
        """
        # Email validation regex pattern - more strict to catch edge cases
        email_pattern = re.compile(
            r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}"
            r"[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+$"
        )
        validators = [
            lambda x: bool(email_pattern.match(x)) and ".." not in x and " " not in x
        ]
        super().__init__(value, validators, "Invalid email format")


class QuantityProperty(ValidatedProperty[float]):
    """A numeric property with units.

    This property stores a quantity with its associated unit.

    Attributes:
        value: The numeric value.
        unit: The unit of measurement.
    """

    def __init__(
        self,
        value: float,
        unit: str,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        """Initialize a quantity property with unit and optional range validation.

        Args:
            value: The numeric value.
            unit: The unit of measurement.
            min_value: Optional minimum allowed value.
            max_value: Optional maximum allowed value.

        Raises:
            ValueError: If the value is outside the allowed range.
        """
        self.unit = unit

        # Create range validators if specified
        validators: list[Callable[[float], bool]] = []
        if min_value is not None:
            validators.append(lambda x: x >= min_value)
        if max_value is not None:
            validators.append(lambda x: x <= max_value)

        # Create error message based on constraints
        error_message = "Invalid quantity"
        if min_value is not None and max_value is not None:
            error_message = f"Value must be between {min_value} and {max_value} {unit}"
        elif min_value is not None:
            error_message = f"Value must be at least {min_value} {unit}"
        elif max_value is not None:
            error_message = f"Value must be at most {max_value} {unit}"

        super().__init__(value, validators, error_message)

    def __str__(self) -> str:
        """Return the string representation with unit."""
        return f"{self.value} {self.unit}"


class SecureProperty(ValidatedProperty[str]):
    """A property that should be handled securely (e.g., API keys).

    This property masks its value in string representations for security.
    """

    def __init__(self, value: str, min_length: int = 1) -> None:
        """Initialize a secure property.

        Args:
            value: The secure string value.
            min_length: Minimum required length.

        Raises:
            ValueError: If the value is too short.
        """
        validators = [lambda x: len(x) >= min_length]
        error_msg = f"Secure value must be at least {min_length} characters"
        super().__init__(value, validators, error_msg)

    def __str__(self) -> str:
        """Return a masked version of the value."""
        if not self.value:
            return ""
        # Show first and last character, mask the rest
        if len(self.value) <= 4:
            return "*" * len(self.value)
        return f"{self.value[0]}{'*' * (len(self.value) - 2)}{self.value[-1]}"

    def __repr__(self) -> str:
        """Return a masked representation."""
        return f"{self.__class__.__name__}('{self.__str__()}')"


# Optional string types with validation
class NonEmptyStringProperty(ValidatedProperty[str]):
    """A string property that cannot be empty."""

    def __init__(self, value: str) -> None:
        """Initialize a non-empty string property.

        Args:
            value: The string value.

        Raises:
            ValueError: If the string is empty.
        """
        validators = [lambda x: len(x.strip()) > 0]
        super().__init__(value, validators, "String cannot be empty")


class LimitedStringProperty(ValidatedProperty[str]):
    """A string property with length limits."""

    def __init__(
        self, value: str, min_length: int = 0, max_length: int | None = None
    ) -> None:
        """Initialize a length-limited string property.

        Args:
            value: The string value.
            min_length: Minimum required length.
            max_length: Optional maximum allowed length.

        Raises:
            ValueError: If the string length is outside the allowed range.
        """
        validators: list[Callable[[str], bool]] = [lambda x: len(x) >= min_length]
        error_message = f"String must be at least {min_length} characters"

        if max_length is not None:
            validators.append(lambda x: len(x) <= max_length)
            error_message = (
                f"String must be between {min_length} and {max_length} characters"
            )

        super().__init__(value, validators, error_message)

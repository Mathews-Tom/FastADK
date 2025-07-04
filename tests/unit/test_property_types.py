"""
Tests for FastADK property types.
"""

import pytest

from fastadk.core.exceptions import ValidationError
from fastadk.core.property_types import (
    EmailProperty,
    LimitedStringProperty,
    NonEmptyStringProperty,
    QuantityProperty,
    SecureProperty,
    URLProperty,
    ValidatedProperty,
)


class TestValidatedProperty:
    """Tests for the ValidatedProperty base class."""

    def test_basic_validation(self):
        """Test that ValidatedProperty performs basic validation."""
        # Valid case
        prop = ValidatedProperty(42, [lambda x: x > 0])
        assert prop.value == 42
        assert prop.validate() is True

        # Invalid case
        with pytest.raises(ValidationError):
            ValidatedProperty(0, [lambda x: x > 0])

    def test_equality_comparison(self):
        """Test equality comparison for ValidatedProperty."""
        prop1 = ValidatedProperty(42)
        prop2 = ValidatedProperty(42)
        prop3 = ValidatedProperty(43)

        assert prop1 == prop2
        assert prop1 != prop3
        assert prop1 == 42
        assert prop1 != 43


class TestURLProperty:
    """Tests for the URLProperty class."""

    def test_valid_urls(self):
        """Test that valid URLs are accepted."""
        valid_urls = [
            "https://example.com",
            "http://example.com/path",
            "http://localhost:8000",
            "https://subdomain.example.co.uk/path?query=value",
            "ftp://ftp.example.org",
            "http://192.168.1.1",
        ]

        for url in valid_urls:
            prop = URLProperty(url)
            assert prop.value == url

    def test_invalid_urls(self):
        """Test that invalid URLs are rejected."""
        invalid_urls = [
            "example.com",  # Missing protocol
            "http:/example.com",  # Missing slash
            "http//example.com",  # Missing colon
            "http:\\\\example.com",  # Backslashes instead of forward slashes
            "example",
            "https://",
            "",
        ]

        for url in invalid_urls:
            with pytest.raises(ValidationError):
                URLProperty(url)


class TestEmailProperty:
    """Tests for the EmailProperty class."""

    def test_valid_emails(self):
        """Test that valid email addresses are accepted."""
        valid_emails = [
            "user@example.com",
            "first.last@example.co.uk",
            "user+tag@example.org",
            "user123@sub.domain.co",
        ]

        for email in valid_emails:
            prop = EmailProperty(email)
            assert prop.value == email

    def test_invalid_emails(self):
        """Test that invalid email addresses are rejected."""
        invalid_emails = [
            "user@",
            "@example.com",
            "user@example",
            "user@.com",
            "user@example..com",
            "user space@example.com",
            "",
        ]

        for email in invalid_emails:
            with pytest.raises(ValidationError):
                EmailProperty(email)


class TestQuantityProperty:
    """Tests for the QuantityProperty class."""

    def test_basic_quantity(self):
        """Test basic quantity with unit."""
        quantity = QuantityProperty(42.5, "kg")
        assert quantity.value == 42.5
        assert quantity.unit == "kg"
        assert str(quantity) == "42.5 kg"

    def test_quantity_range_validation(self):
        """Test quantity with range validation."""
        # Valid cases
        QuantityProperty(5, "m", min_value=0)
        QuantityProperty(5, "m", max_value=10)
        QuantityProperty(5, "m", min_value=0, max_value=10)

        # Invalid cases
        with pytest.raises(ValidationError):
            QuantityProperty(-1, "m", min_value=0)

        with pytest.raises(ValidationError):
            QuantityProperty(11, "m", max_value=10)

        with pytest.raises(ValidationError):
            QuantityProperty(15, "m", min_value=0, max_value=10)


class TestSecureProperty:
    """Tests for the SecureProperty class."""

    def test_secure_property_masking(self):
        """Test that SecureProperty masks its value in string representation."""
        # Short value
        short = SecureProperty("123")
        assert str(short) == "***"

        # Longer value
        api_key = SecureProperty("api_key_12345")
        assert str(api_key) == "a***********5"
        assert api_key.value == "api_key_12345"

        # Representation should also be masked
        assert "*" in repr(api_key)

    def test_minimum_length_validation(self):
        """Test minimum length validation for SecureProperty."""
        # Valid case
        SecureProperty("12345", min_length=5)

        # Invalid case
        with pytest.raises(ValidationError):
            SecureProperty("123", min_length=5)


class TestStringProperties:
    """Tests for string property types."""

    def test_non_empty_string(self):
        """Test that NonEmptyStringProperty rejects empty strings."""
        # Valid cases
        NonEmptyStringProperty("hello")
        NonEmptyStringProperty("  hello  ")  # Whitespace is trimmed in validation

        # Invalid cases
        with pytest.raises(ValidationError):
            NonEmptyStringProperty("")

        with pytest.raises(ValidationError):
            NonEmptyStringProperty("   ")  # Just whitespace

    def test_limited_string(self):
        """Test that LimitedStringProperty enforces length constraints."""
        # Valid cases
        LimitedStringProperty("hello", min_length=1)
        LimitedStringProperty("hello", min_length=1, max_length=10)

        # Invalid cases - too short
        with pytest.raises(ValidationError):
            LimitedStringProperty("", min_length=1)

        # Invalid cases - too long
        with pytest.raises(ValidationError):
            LimitedStringProperty("hello world", min_length=1, max_length=5)

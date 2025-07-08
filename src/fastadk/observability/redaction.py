"""
Redaction middleware for FastADK.

This module provides redaction capabilities for sensitive data in logs and traces.
"""

import re
from typing import Any, Dict, List, Optional, Pattern

from fastadk.observability.logger import logger


class RedactionFilter:
    """Filter for redacting sensitive data in logs and traces."""

    # Default patterns to redact
    DEFAULT_PATTERNS = [
        # API keys and tokens
        r'(api[_-]?key|apikey|token|secret|password|auth)["\']?\s*[=:]\s*["\']?([^\s"\']{8,})["\']?',
        # Credit card numbers
        r"\b(?:\d{4}[- ]?){3}\d{4}\b",
        # SSNs
        r"\b\d{3}[-]?\d{2}[-]?\d{4}\b",
        # Email addresses
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        # Phone numbers
        r"\b(?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
    ]

    def __init__(
        self,
        enabled: bool = True,
        patterns: Optional[List[str]] = None,
        replacement: str = "[REDACTED]",
    ) -> None:
        """Initialize the redaction filter.

        Args:
            enabled: Whether redaction is enabled
            patterns: List of regex patterns to redact
            replacement: The string to replace sensitive data with
        """
        self.enabled = enabled
        self.patterns = patterns or self.DEFAULT_PATTERNS
        self.replacement = replacement
        self._compiled_patterns: List[Pattern[str]] = []
        self._compile_patterns()
        logger.info(
            "Redaction filter initialized",
            enabled=enabled,
            pattern_count=len(self.patterns),
        )

    def _compile_patterns(self) -> None:
        """Compile the regex patterns."""
        self._compiled_patterns = [re.compile(pattern) for pattern in self.patterns]

    def add_pattern(self, pattern: str) -> None:
        """Add a new pattern to redact.

        Args:
            pattern: The regex pattern to add
        """
        self.patterns.append(pattern)
        self._compiled_patterns.append(re.compile(pattern))
        logger.info("Added redaction pattern", pattern=pattern)

    def redact(self, text: str) -> str:
        """Redact sensitive information from text.

        Args:
            text: The text to redact

        Returns:
            The redacted text
        """
        if not self.enabled:
            return text

        redacted = text
        for pattern in self._compiled_patterns:
            # For patterns with capturing groups, only redact the sensitive part
            if "(" in pattern.pattern and ")" in pattern.pattern:
                try:
                    # Try to keep the field name and replace just the value
                    redacted = pattern.sub(
                        lambda m: (
                            m.group(1) + "=" + self.replacement
                            if len(m.groups()) > 0
                            else self.replacement
                        ),
                        redacted,
                    )
                except Exception:
                    # Fall back to simple replacement
                    redacted = pattern.sub(self.replacement, redacted)
            else:
                redacted = pattern.sub(self.replacement, redacted)

        return redacted

    def redact_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive information from a dictionary.

        Args:
            data: The dictionary to redact

        Returns:
            The redacted dictionary
        """
        if not self.enabled:
            return data

        result = {}
        for key, value in data.items():
            # Check if the key suggests sensitive data
            if any(
                k in key.lower()
                for k in ["api_key", "token", "secret", "password", "auth"]
            ):
                result[key] = self.replacement
            # Recursively redact nested dictionaries
            elif isinstance(value, dict):
                result[key] = self.redact_dict(value)  # type: ignore
            # Redact strings
            elif isinstance(value, str):
                result[key] = self.redact(value)
            # Handle lists with potential sensitive data
            elif isinstance(value, list):
                result[key] = [
                    (
                        self.redact_dict(item)  # type: ignore
                        if isinstance(item, dict)
                        else self.redact(item) if isinstance(item, str) else item
                    )
                    for item in value
                ]  # type: ignore
            else:
                result[key] = value

        return result


# Create a global instance of the redaction filter
redaction = RedactionFilter()

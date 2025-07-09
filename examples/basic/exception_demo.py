"""
Exception handling demonstration for FastADK.

This example demonstrates the comprehensive exception handling system
and shows how different types of errors are captured, translated, and
presented to users in a consistent way.
"""

import asyncio
import logging
from typing import Any

import requests

from fastadk.core.agent import Agent, BaseAgent, tool
from fastadk.core.exceptions import (
    ConfigurationError,
    ExceptionTranslator,
    FastADKError,
    ToolError,
    ValidationError,
)
from fastadk.core.property_types import EmailProperty, QuantityProperty, URLProperty

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@Agent(
    model="gemini-2.5-flash",
    description="An agent demonstrating exception handling",
)
class ExceptionDemoAgent(BaseAgent):
    """Agent that demonstrates the various exception handling capabilities of FastADK."""

    @tool
    def validate_user(self, email: str, age: int) -> Any:
        """
        Validate user information with robust error handling.

        Args:
            email: User's email address
            age: User's age (must be 18 or older)

        Returns:
            Dictionary with validation result
        """
        try:
            # Use property types for validation
            email_prop = EmailProperty(email)
            age_prop = QuantityProperty(age, "years", min_value=18)

            return {"status": "valid", "email": str(email_prop), "age": str(age_prop)}
        except ValidationError as e:
            # The error already has proper error code and details
            raise e
        except ValueError as e:
            # Translate standard Python exceptions to FastADK exceptions
            raise ExceptionTranslator.translate_exception(
                e,
                default_message="User validation failed",
                default_error_code="USER_VALIDATION_ERROR",
            ) from e

    @tool
    def fetch_external_data(self, url: str) -> Any:
        """
        Fetch data from an external API with proper error handling.

        Args:
            url: The URL to fetch data from

        Returns:
            Dictionary with the fetched data
        """
        try:
            # Validate URL
            url_prop = URLProperty(url)

            # Make the external request
            response = requests.get(str(url_prop), timeout=5)
            response.raise_for_status()

            result = response.json()
            return result
        except requests.exceptions.RequestException as e:
            # Automatically translate request exceptions
            raise ExceptionTranslator.translate_exception(e) from e
        except Exception as e:
            # Fallback for any other unexpected errors
            raise ToolError(
                message=f"Failed to fetch data: {str(e)}",
                error_code="FETCH_ERROR",
                details={"url": url},
            ) from e

    @tool
    def check_configuration(self, config_type: str) -> Any:
        """
        Check configuration settings with proper error handling.

        Args:
            config_type: Type of configuration to check (api, database, security)

        Returns:
            Dictionary with configuration status
        """
        valid_types = ["api", "database", "security"]

        if config_type not in valid_types:
            raise ConfigurationError(
                message=f"Invalid configuration type: {config_type}",
                error_code="INVALID_CONFIG_TYPE",
                details={"provided": config_type, "valid_options": valid_types},
            )

        # Simulate configuration check
        if config_type == "api":
            return {"status": "valid", "message": "API configuration is valid"}
        elif config_type == "database":
            # Simulate a database connection issue
            raise ConfigurationError(
                message="Database configuration is invalid",
                error_code="DB_CONFIG_ERROR",
                details={
                    "missing_fields": ["db_password"],
                    "environment": "development",
                },
            )
        else:
            return {"status": "valid", "message": "Security configuration is valid"}


async def main() -> None:
    """Run the demo agent with various exception scenarios."""
    agent = ExceptionDemoAgent()

    print("\n🚀 Exception Handling Demo Agent\n")

    # Test 1: Validation error
    print("\n🔍 Testing email validation...")
    try:
        result = await agent.execute_tool(
            "validate_user", email="invalid-email", age=25
        )
        print(f"✅ Result: {result}")
    except FastADKError as e:
        print(f"❌ Error [{e.error_code}]: {e.message}")
        print(f"   Details: {e.details}")

    # Test 2: Age validation
    print("\n🔍 Testing age validation...")
    try:
        result = await agent.execute_tool(
            "validate_user", email="user@example.com", age=16
        )
        print(f"✅ Result: {result}")
    except FastADKError as e:
        print(f"❌ Error [{e.error_code}]: {e.message}")
        print(f"   Details: {e.details}")

    # Test 3: External request error
    print("\n🔍 Testing external API error handling...")
    try:
        result = await agent.execute_tool(
            "fetch_external_data", url="https://non-existent-url.example.com"
        )
        print(f"✅ Result: {result}")
    except FastADKError as e:
        print(f"❌ Error [{e.error_code}]: {e.message}")
        print(f"   Details: {e.details}")

    # Test 4: Configuration error
    print("\n🔍 Testing configuration error handling...")
    try:
        result = await agent.execute_tool("check_configuration", config_type="invalid")
        print(f"✅ Result: {result}")
    except FastADKError as e:
        print(f"❌ Error [{e.error_code}]: {e.message}")
        print(f"   Details: {e.details}")

    # Test 5: Successful validation
    print("\n🔍 Testing successful validation...")
    try:
        result = await agent.execute_tool(
            "validate_user", email="user@example.com", age=25
        )
        print(f"✅ Result: {result}")
    except FastADKError as e:
        print(f"❌ Error [{e.error_code}]: {e.message}")

    # Test 6: Successful configuration check
    print("\n🔍 Testing successful configuration check...")
    try:
        result = await agent.execute_tool("check_configuration", config_type="api")
        print(f"✅ Result: {result}")
    except FastADKError as e:
        print(f"❌ Error [{e.error_code}]: {e.message}")


if __name__ == "__main__":
    asyncio.run(main())

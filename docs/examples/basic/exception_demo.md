# Exception Handling Demo

This example demonstrates FastADK's comprehensive exception handling system, showing how different types of errors are captured, translated, and presented to users in a consistent way.

## Features Demonstrated

- Property type validation with custom validation rules
- Converting standard Python exceptions to FastADK exceptions
- Error code and details standardization
- Handling external API errors
- Configuration validation and error handling
- Using FastADK's exception hierarchy

## Prerequisites

To run this example, you need:

```bash
# Install the requests library
uv add requests
```

No API key is required for this example.

## How It Works

The example includes an `ExceptionDemoAgent` with three tools that demonstrate different aspects of exception handling:

1. `validate_user`: Demonstrates property validation for email and age
   - Uses `EmailProperty` to validate email format
   - Uses `QuantityProperty` to validate age with a minimum value constraint

2. `fetch_external_data`: Shows handling of external API errors
   - Validates URLs using `URLProperty`
   - Translates standard request exceptions to FastADK exceptions
   - Includes fallback handling for unexpected errors

3. `check_configuration`: Demonstrates configuration validation
   - Shows custom error codes and details
   - Validates input against a predefined list of valid options

The example runs through various test cases that trigger different types of exceptions, showing how they are caught, processed, and displayed to the user.

## Expected Output

When you run the script, you'll see output similar to:

```bash
🚀 Exception Handling Demo Agent

🔍 Testing email validation...
❌ Error [TOOL_EXECUTION_ERROR]: Tool 'validate_user' failed: [PROPERTY_VALIDATION_FAILED] Invalid email format: invalid-email
   Details: {'tool_name': 'validate_user', 'original_error': '[PROPERTY_VALIDATION_FAILED] Invalid email format: invalid-email', 'error_type': 'ValidationError'}

🔍 Testing age validation...
❌ Error [TOOL_EXECUTION_ERROR]: Tool 'validate_user' failed: [PROPERTY_VALIDATION_FAILED] Value must be at least 18 years: 16
   Details: {'tool_name': 'validate_user', 'original_error': '[PROPERTY_VALIDATION_FAILED] Value must be at least 18 years: 16', 'error_type': 'ValidationError'}

🔍 Testing external API error handling...
❌ Error [TOOL_EXECUTION_ERROR]: Tool 'fetch_external_data' failed: [EXTERNAL_CONNECTIONERROR] External error: HTTPSConnectionPool(host='non-existent-url.example.com', port=443): Max retries exceeded
   Details: {'tool_name': 'fetch_external_data', 'original_error': '[EXTERNAL_CONNECTIONERROR] External error...', 'error_type': 'ServiceUnavailableError'}

🔍 Testing configuration error handling...
❌ Error [TOOL_EXECUTION_ERROR]: Tool 'check_configuration' failed: [INVALID_CONFIG_TYPE] Invalid configuration type: invalid
   Details: {'tool_name': 'check_configuration', 'original_error': '[INVALID_CONFIG_TYPE] Invalid configuration type: invalid', 'error_type': 'ConfigurationError'}

🔍 Testing successful validation...
✅ Result: {'status': 'valid', 'email': 'user@example.com', 'age': '25 years'}

🔍 Testing successful configuration check...
✅ Result: {'status': 'valid', 'message': 'API configuration is valid'}
```

## Key Concepts

1. **Error Hierarchy**: FastADK has a hierarchy of exception classes that help categorize errors.

2. **Error Translation**: The `ExceptionTranslator` converts standard Python exceptions to FastADK exceptions.

3. **Standardized Error Format**: All FastADK errors include:
   - A human-readable message
   - A machine-readable error code
   - A details dictionary with contextual information

4. **Property Validation**: FastADK's property types provide built-in validation with meaningful error messages.

## Best Practices Demonstrated

- Using specific exception types for different error categories
- Including detailed error context in exception details
- Handling and translating external API errors
- Validating inputs before processing
- Providing clear, user-friendly error messages

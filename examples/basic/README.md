# Basic FastADK Examples

This directory contains basic examples that demonstrate core FastADK functionality.

## Weather Agent Example

The `weather_agent.py` example demonstrates:

- Using the `@Agent` and `@tool` decorators
- Real API integration with wttr.in
- Asynchronous tool implementation
- Loading system prompts from files
- Type hinting for tool parameters

### Setup and Running

1. Install required dependencies:

   ```bash
   uv add httpx python-dotenv
   ```

2. Set up your API key either via environment variable:

   ```bash
   export GEMINI_API_KEY=your_api_key_here
   ```

   Or in a `.env` file:

   ```bash
   GEMINI_API_KEY=your_api_key_here
   ```

3. Run the example:

   ```bash
   uv run weather_agent.py
   ```

### Expected Output

```bash
INFO:fastadk.agent:Initialized Gemini model gemini-1.5-pro
INFO:fastadk.agent:Initialized agent WeatherAgent with 2 tools

--- Testing Agent with a sample query ---
INFO:__main__:WeatherAgent LIVE processing starting
INFO:__main__:WeatherAgent LIVE response length: 521
INFO:fastadk.agent:Agent execution completed in 4.19s

Final Response:
The current weather in London is cloudy with a temperature of 18°C (64°F).
It feels like 17°C (63°F) due to a light breeze and 75% humidity.

I'd recommend bringing a light jacket or sweater as it might feel a bit cool,
especially if you'll be outside for extended periods. An umbrella isn't
necessary as there's no rain in the immediate forecast.

The forecast for the next three days shows a gradual warming trend with
temperatures rising to 21°C by Friday, with partly cloudy conditions expected.
```

## Exception Demo Example

The `exception_demo.py` example demonstrates:

- Comprehensive exception handling
- Property type validation
- Converting standard Python exceptions to FastADK exceptions
- Error code and details standardization

### Setup and Running

1. Install required dependencies:

   ```bash
   uv add requests
   ```

2. Run the example:

   ```bash
   uv run exception_demo.py
   ```

### Expected Output

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

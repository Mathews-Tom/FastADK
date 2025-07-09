# Tool Development Patterns

This example demonstrates different patterns for developing and using tools in FastADK, showing various approaches to tool implementation, validation, error handling, and registration.

## Features Demonstrated

- Basic synchronous and asynchronous tools
- Tools with validation and error handling using Pydantic
- Tools with nested structure and rich returns
- Parameterized tools with default values
- Dynamic tool registration patterns
- Tool composition and reuse

## Prerequisites

To run this example:

```bash
uv add python-dotenv
uv run tool_patterns.py
```

No API key is required as the example works with the simulated provider.

## How It Works

The example creates a `ToolPatternsAgent` that demonstrates six different patterns for tool development:

### Pattern 1: Basic Synchronous Tool

The `get_current_time` tool shows a simple synchronous tool that returns structured time information:

- No input parameters
- Dictionary return type
- Simple implementation

### Pattern 2: Async Tool with Simulated Delay

The `get_random_number` tool demonstrates:

- Asynchronous implementation with `async/await`
- Parameter defaults (min_value=1, max_value=100)
- Simulated delay to demonstrate async operation
- Structured return with multiple fields

### Pattern 3: Tool with Pydantic Validation

The `get_weather` tool shows:

- Input validation using Pydantic models
- Field validators with custom logic
- Structured error handling for validation failures
- Normalized input processing

### Pattern 4: Tool with Rich Error Handling

The `search` tool demonstrates:

- Comprehensive error handling strategies
- Performance timing
- Structured response objects
- Try/except patterns for graceful failure

### Pattern 5: Nested Tool with Hierarchical Structure

The `get_system_info` tool shows:

- Complex nested data structures
- Hierarchical information organization
- Multiple levels of nesting

### Pattern 6: Dynamic Tool Registration

This pattern shows how to:

- Define tools programmatically at runtime
- Register tools based on runtime conditions
- Add both synchronous and asynchronous tools dynamically

## Data Models

The example includes several Pydantic models for request and response validation:

- `WeatherRequest`: Validates location and units for weather requests
- `WeatherResponse`: Structures weather information responses
- `SearchRequest`: Validates search queries and result limits
- `SearchResult`: Models individual search results
- `SearchResponse`: Structures complete search responses

## Expected Output

When you run the script, you'll see output explaining the different tool patterns:

```bash
============================================================
🛠️  FastADK Tool Patterns Demo
============================================================

🚀 Initializing agent...
Would register calculator tool here
Would register convert_currency tool here
  (Note: Dynamic tools would be added here)

📋 Available tools: 5


📌 PATTERN 1: BASIC SYNCHRONOUS TOOL
------------------------------------------------------------
Tool: get_current_time
Description: Returns the current time and date in a formatted structure


📌 PATTERN 2: ASYNC TOOL WITH SIMULATED DELAY
------------------------------------------------------------
Tool: get_random_number
Description: Generates a random number in a specified range with a simulated delay


📌 PATTERN 3: TOOL WITH PYDANTIC VALIDATION
------------------------------------------------------------
Tool: get_weather
Description: Gets weather data for a location with input validation
Benefits: Automatic validation of input parameters and helpful error messages


📌 PATTERN 4: TOOL WITH RICH ERROR HANDLING
------------------------------------------------------------
Tool: search
Description: Demonstrates comprehensive error handling with detailed error reporting


📌 PATTERN 5: NESTED TOOL WITH HIERARCHICAL STRUCTURE
------------------------------------------------------------
Tool: get_system_info
Description: Shows how to structure complex nested data in tool responses


📌 PATTERN 6: DYNAMICALLY REGISTERED TOOLS
------------------------------------------------------------
Tool: calculator and convert_currency
Description: Shows how to dynamically register tools at runtime
Examples:
  - calculator: Perform mathematical operations (add, subtract, multiply, divide)
  - convert_currency: Convert between different currencies

📝 Summary of Tool Patterns Demonstrated:
1. Basic synchronous tool - Simple operation with return value
2. Async tool - Support for asynchronous operations
3. Tool with Pydantic validation - Type and value validation
4. Tool with rich error handling - Detailed error reporting
5. Tool with hierarchical data - Structured nested responses
6. Dynamically registered tools - Runtime tool registration

============================================================
🏁 FastADK -  Tool Patterns Demo Completed
============================================================
```

## Key Concepts

1. **Tool Decoration**: Using the `@tool` decorator to expose methods as tools with appropriate type hints and documentation.

2. **Pydantic Integration**: Leveraging Pydantic models for request and response validation.

3. **Async Support**: Building tools that can operate asynchronously for non-blocking operations.

4. **Error Handling**: Implementing robust error handling with structured error responses.

5. **Dynamic Registration**: Adding tools programmatically at runtime based on application needs.

## Best Practices Demonstrated

- Using descriptive docstrings for tool documentation
- Implementing proper error handling and validation
- Structuring complex data in logical hierarchies
- Providing default values for optional parameters
- Using asynchronous functions for operations that might block
- Validating inputs before processing
- Including timing information for performance monitoring

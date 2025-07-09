# Tools in FastADK

Tools are the primary way for agents to interact with external systems, data sources, and services. This document explains how tools work in FastADK, how to create them, and best practices for tool development.

## What Are Tools?

In the context of FastADK, a tool is a function that an agent can call to perform a specific task. Tools can:

- Fetch data from APIs or databases
- Perform calculations
- Manipulate data
- Interact with external systems
- Execute business logic

Tools bridge the gap between the language model's reasoning capabilities and the ability to take actions in the real world.

## Creating Tools

FastADK makes it easy to create tools using the `@tool` decorator:

```python
from fastadk import Agent, BaseAgent, tool
from typing import Dict, List

@Agent(model="gemini-1.5-pro")
class WeatherAgent(BaseAgent):
    @tool
    def get_weather(self, city: str, country: str = "US") -> Dict:
        """
        Get the current weather for a city.
        
        Args:
            city: The name of the city
            country: The country code (default: "US")
            
        Returns:
            A dictionary containing weather information
        """
        # Tool implementation goes here
        return {
            "city": city,
            "country": country,
            "temperature": 22.5,
            "condition": "Sunny",
            "humidity": 65
        }
```

## Tool Features

### Type Hints

FastADK uses Python type hints to validate inputs and outputs. This helps prevent errors and provides better documentation.

```python
@tool
def calculate_area(length: float, width: float) -> float:
    """Calculate the area of a rectangle."""
    return length * width
```

### Default Parameters

Tools can have default parameters to make them more flexible:

```python
@tool
def search_products(
    query: str,
    category: str = "all",
    max_results: int = 10,
    sort_by: str = "relevance"
) -> List[Dict]:
    """Search for products in our catalog."""
    # Implementation
```

### Documentation

Good documentation is crucial for tools. The LLM uses the docstring to understand when and how to use the tool. Include:

- A clear description of what the tool does
- Parameter descriptions
- Return value information
- Examples if needed

### Advanced Configuration

The `@tool` decorator accepts several configuration options:

```python
@tool(
    name="fetch_data",               # Custom name (defaults to function name)
    description="Fetch data from API", # Override docstring description
    required=["api_key", "query"],   # Required parameters
    cache_ttl=300,                   # Cache results for 5 minutes
    retry=3,                         # Retry up to 3 times on failure
    timeout=10,                      # Timeout after 10 seconds
    rate_limit=100                   # Max calls per minute
)
def fetch_data_from_api(api_key: str, query: str) -> Dict:
    """Implementation details..."""
```

## Tool Categories

### Data Retrieval Tools

These tools fetch information from databases, APIs, or other data sources.

```python
@tool
async def search_database(query: str) -> List[Dict]:
    """Search the database for records matching the query."""
    # Database connection and query logic
```

### Computational Tools

These tools perform calculations or transformations on data.

```python
@tool
def analyze_sentiment(text: str) -> Dict:
    """Analyze the sentiment of the provided text."""
    # Sentiment analysis logic
```

### Integration Tools

These tools integrate with external systems and APIs.

```python
@tool
async def send_email(to: str, subject: str, body: str) -> bool:
    """Send an email to the specified recipient."""
    # Email sending logic
```

### User Interaction Tools

These tools facilitate interaction with users.

```python
@tool
def format_response(data: Dict, format: str = "text") -> str:
    """Format data for presentation to the user."""
    # Formatting logic
```

## Best Practices

### Make Tools Atomic

Each tool should do one thing well. Instead of a single `manage_user` tool, consider separate tools for `create_user`, `update_user`, `delete_user`, etc.

### Validate Inputs

Even though FastADK provides basic type validation, consider adding additional validation in your tool implementation:

```python
@tool
def transfer_money(from_account: str, to_account: str, amount: float) -> Dict:
    """Transfer money between accounts."""
    if amount <= 0:
        raise ValueError("Amount must be positive")
    if from_account == to_account:
        raise ValueError("Cannot transfer to the same account")
    # Transfer logic
```

### Handle Errors Gracefully

Use FastADK's error handling to provide informative error messages:

```python
from fastadk.core.exceptions import ToolError

@tool
def fetch_data(url: str) -> Dict:
    """Fetch data from a URL."""
    try:
        # Fetch logic
    except ConnectionError:
        raise ToolError("Could not connect to the server. Please check the URL or try again later.")
    except Exception as e:
        raise ToolError(f"An error occurred: {str(e)}")
```

### Provide Context in Responses

Include contextual information in your tool responses to help the agent understand the results:

```python
@tool
def search_products(query: str, max_results: int = 5) -> Dict:
    """Search for products in the catalog."""
    results = # Search logic
    
    return {
        "query": query,
        "total_matches": len(all_results),
        "showing": min(len(all_results), max_results),
        "results": results[:max_results]
    }
```

### Document Side Effects

If your tool has side effects (like modifying a database or sending an email), clearly document this in the docstring:

```python
@tool
def create_order(product_id: str, quantity: int, customer_id: str) -> Dict:
    """
    Create a new order in the system.
    
    Note: This tool will charge the customer's account and initiate shipping.
    """
    # Order creation logic
```

## Tool Patterns

### Chaining Tools

Complex operations can be broken down into multiple tools that can be chained together:

```python
@Agent(model="gemini-1.5-pro")
class OrderProcessor(BaseAgent):
    @tool
    def check_inventory(self, product_id: str) -> Dict:
        """Check if a product is in stock."""
        # Inventory check logic
    
    @tool
    def calculate_price(self, product_id: str, quantity: int) -> Dict:
        """Calculate the total price including taxes and shipping."""
        # Price calculation logic
    
    @tool
    def process_payment(self, customer_id: str, amount: float) -> Dict:
        """Process payment for an order."""
        # Payment processing logic
    
    @tool
    def create_order(self, product_id: str, quantity: int, customer_id: str) -> Dict:
        """Create a new order after inventory and payment are confirmed."""
        # Order creation logic
```

### Tool Composition

You can create higher-level tools that use other tools:

```python
@tool
async def complete_purchase(self, product_id: str, quantity: int, customer_id: str) -> Dict:
    """Process a complete purchase flow."""
    # First check inventory
    inventory = await self.check_inventory(product_id)
    if not inventory["in_stock"] or inventory["quantity"] < quantity:
        return {"success": False, "reason": "Not enough inventory"}
    
    # Calculate price
    price_info = await self.calculate_price(product_id, quantity)
    
    # Process payment
    payment = await self.process_payment(customer_id, price_info["total"])
    if not payment["success"]:
        return {"success": False, "reason": f"Payment failed: {payment['message']}"}
    
    # Create order
    order = await self.create_order(product_id, quantity, customer_id)
    
    return {
        "success": True,
        "order_id": order["id"],
        "total": price_info["total"],
        "estimated_delivery": order["estimated_delivery"]
    }
```

## Conclusion

Tools are the primary way for FastADK agents to interact with the world and perform useful tasks. By designing tools carefully and following best practices, you can create agents that are capable of solving complex problems by combining reasoning with action.

For practical examples of tool implementations, see the [Examples](../examples/) section.

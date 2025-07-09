# Chain of Thought Reasoning Demo

This example demonstrates how to create a FastADK agent that shows its reasoning process, making tool selection and thought process transparent to users.

## Features Demonstrated

- Chain of thought reasoning with visible thought process
- Tool selection process transparency
- Different types of tools and their integration
- Graceful handling of API key availability
- Using the `show_reasoning=True` parameter

## Prerequisites

This example works without any API keys using the simulated provider. However, for the best experience with real responses, set up a Gemini API key:

```bash
export GEMINI_API_KEY=your_api_key_here
```

To run the example:

```bash
uv run reasoning_demo.py
```

## How It Works

The `ReasoningAgent` demonstrates transparent reasoning by:

1. Using the `show_reasoning=True` parameter to expose the agent's thought process
2. Implementing a variety of tools that require different reasoning approaches:
   - `get_planet_facts`: Retrieves facts about planets from a database
   - `solve_math_problem`: Solves mathematical problems
   - `check_weather`: Provides simulated weather information
   - `search_database`: Performs a search across a simulated database

The agent processes different types of queries that demonstrate how it reasons through problems, selects appropriate tools, and formulates responses based on the tools' outputs.

## Example Queries

The demo runs through five different types of queries:

1. **Factual information**: "Can you tell me some facts about Mars?"
2. **Weather information with reasoning**: "What's the weather like in Tokyo and should I bring an umbrella?"
3. **Math problem solving**: "I need to solve this math problem: What is 25 × 13?"
4. **Database search**: "Search for information about moons in our solar system"
5. **Comparative analysis**: "Compare Earth and Jupiter based on their size and composition"

## Expected Output

When you run the script, you'll see output similar to:

```bash
==================================================
🧠 FastADK Chain of Thought Reasoning Demo 🧠
==================================================

📌 Example 1/5

💬 User Query: Can you tell me some facts about Mars?

🤖 Processing...

📝 Final Response:
I need to provide facts about Mars. Let me use the get_planet_facts tool to retrieve this information.

Thinking through the steps:
1. The user wants facts about Mars
2. I have a tool called get_planet_facts that can retrieve planet information
3. I'll use this tool with "mars" as the parameter

Using get_planet_facts tool with planet="mars"...

The tool returned:
{
  "planet": "mars",
  "found": true,
  "facts": [
    "Mars is the fourth planet from the Sun",
    "Mars has a thin atmosphere composed primarily of carbon dioxide",
    "Mars has two small moons, Phobos and Deimos",
    "Mars is often called the 'Red Planet'",
    "Mars has the largest volcano in the solar system, Olympus Mons"
  ]
}

Here are some facts about Mars:

1. Mars is the fourth planet from the Sun
2. Mars has a thin atmosphere composed primarily of carbon dioxide
3. Mars has two small moons, Phobos and Deimos
4. Mars is often called the 'Red Planet'
5. Mars has the largest volcano in the solar system, Olympus Mons

🔧 Tools Used:
  - get_planet_facts
```

The response includes:

- The initial query
- The agent's step-by-step reasoning process
- The tool selected and parameters used
- The raw output from the tool
- The final formatted response
- A list of tools used during processing

## Key Concepts

1. **Transparent Reasoning**: By enabling `show_reasoning=True`, the agent exposes its internal thought process, helping users understand how it arrived at its conclusions.

2. **Tool Selection Logic**: The agent demonstrates how it selects appropriate tools based on the query, showing the criteria it uses to match user needs with available capabilities.

3. **Graceful Degradation**: The example works even without an API key by using simulated responses, demonstrating graceful handling of missing credentials.

4. **Structured Tool Responses**: Each tool returns structured data (dictionaries) that the agent can reason about and transform into natural language responses.

## Best Practices Demonstrated

- Using structured documentation in tool methods
- Providing clear error messages when tools can't fulfill requests
- Separating data retrieval (tools) from response generation (agent)
- Making reasoning transparent to build user trust
- Using type hints for better code clarity and tool parameter handling

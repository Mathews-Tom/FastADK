# Multi-Provider Reasoning Example

This example demonstrates how to build an agent that can dynamically use different LLM providers based on available API keys and fallback strategies. It showcases FastADK's provider-agnostic design and ability to handle complex reasoning tasks across different models.

## Overview

The Multi-Provider Reasoning agent is designed to:

1. Check for available API keys for different providers (OpenAI, Gemini, Anthropic)
2. Select the best available provider based on priority and availability
3. Fall back to alternative providers if the primary one fails
4. Adapt prompts and parsing strategies based on the selected provider
5. Perform consistent reasoning tasks regardless of the underlying model

## Key Features

### Dynamic Provider Selection

The agent dynamically selects a provider based on available API keys:

```python
def select_provider(self):
    """Select the best available provider based on API key availability."""
    providers = {
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": "gpt-4",
            "priority": 1
        },
        "gemini": {
            "api_key": os.getenv("GEMINI_API_KEY"),
            "model": "gemini-1.5-pro",
            "priority": 2
        },
        "anthropic": {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": "claude-3-opus-20240229",
            "priority": 3
        }
    }
    
    # Filter to providers with API keys
    available_providers = {k: v for k, v in providers.items() if v["api_key"]}
    
    if not available_providers:
        raise ValueError("No API keys found for any provider")
    
    # Return the highest priority (lowest number) provider
    return min(available_providers.items(), key=lambda x: x[1]["priority"])
```

### Provider-Specific Adaptations

The agent adapts its behavior based on the selected provider:

```python
def on_initialize(self):
    """Initialize the agent with the best available provider."""
    provider_name, provider_config = self.select_provider()
    
    # Set agent properties based on selected provider
    self.provider_name = provider_name
    self.model_name = provider_config["model"]
    
    # Adjust prompt templates based on provider
    if provider_name == "anthropic":
        self.system_prompt = self.system_prompt_anthropic
    elif provider_name == "gemini":
        self.system_prompt = self.system_prompt_gemini
    else:
        self.system_prompt = self.system_prompt_openai
```

### Reasoning Tools

The agent implements tools for complex reasoning tasks:

```python
@tool
def analyze_argument(self, argument: str) -> Dict[str, Any]:
    """
    Analyze the logical structure and validity of an argument.
    
    Args:
        argument: The argument text to analyze
        
    Returns:
        Dictionary containing analysis of premises, conclusion, and logical validity
    """
    # Implementation details...

@tool
def identify_cognitive_biases(self, text: str) -> List[Dict[str, str]]:
    """
    Identify potential cognitive biases in the provided text.
    
    Args:
        text: The text to analyze for cognitive biases
        
    Returns:
        List of identified biases with explanations
    """
    # Implementation details...

@tool
def evaluate_evidence(self, claim: str, evidence: List[str]) -> Dict[str, Any]:
    """
    Evaluate the strength of evidence supporting a claim.
    
    Args:
        claim: The claim being evaluated
        evidence: List of evidence points supporting the claim
        
    Returns:
        Evaluation of evidence strength and overall confidence in the claim
    """
    # Implementation details...
```

### Fallback Strategies

The agent implements fallback strategies for handling provider failures:

```python
@retry(max_attempts=3, backoff_factor=2)
async def run_with_fallback(self, prompt: str) -> str:
    """Run the agent with automatic fallback to alternative providers."""
    try:
        return await super().run(prompt)
    except Exception as e:
        logger.warning(f"Error with provider {self.provider_name}: {str(e)}")
        
        # Try to switch providers
        available_providers = self.get_alternative_providers()
        if not available_providers:
            raise RuntimeError("All providers failed")
            
        # Select next best provider
        self.provider_name, provider_config = available_providers[0]
        self.model_name = provider_config["model"]
        
        # Adjust system prompt
        self.update_system_prompt()
        
        # Retry with new provider
        return await super().run(prompt)
```

## Implementation Details

### Provider-Specific System Prompts

Different providers may require slightly different system prompt formats:

```python
system_prompt_openai = """
You are an advanced reasoning assistant specialized in critical thinking, 
logical analysis, and identifying cognitive biases.
"""

system_prompt_anthropic = """
<instructions>
You are an advanced reasoning assistant specialized in critical thinking, 
logical analysis, and identifying cognitive biases.
</instructions>
"""

system_prompt_gemini = """
You are an advanced reasoning assistant specialized in critical thinking, 
logical analysis, and identifying cognitive biases.
"""
```

### Provider Configuration

The agent uses environment variables for API keys:

```python
# .env file example
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=sk-ant-...
```

### Response Parsing Adaptations

The agent adapts its response parsing based on the provider:

```python
def parse_response(self, response: str) -> Dict[str, Any]:
    """Parse the response based on the current provider."""
    if self.provider_name == "anthropic":
        # Claude may format outputs differently
        return self.parse_anthropic_response(response)
    elif self.provider_name == "gemini":
        # Gemini may have its own output format
        return self.parse_gemini_response(response)
    else:
        # Default parsing for OpenAI
        return self.parse_openai_response(response)
```

## Usage Example

Here's how you might use the Multi-Provider Reasoning agent:

```python
from multi_provider_reasoning import ReasoningAgent

async def main():
    agent = ReasoningAgent()
    
    # The agent will automatically select the best available provider
    
    # Analyze an argument
    response = await agent.run(
        "Analyze this argument: 'All humans are mortal. Socrates is human. Therefore, Socrates is mortal.'"
    )
    print(response)
    
    # Identify cognitive biases
    response = await agent.run(
        "Identify cognitive biases in this text: 'I won the lottery last week after wearing my lucky socks, so now I always wear them when buying tickets.'"
    )
    print(response)
    
    # Evaluate evidence
    response = await agent.run(
        "Evaluate the evidence for this claim: 'Coffee consumption reduces the risk of type 2 diabetes.' Evidence: 1) A study of 120,000 people showed 30% lower risk in regular coffee drinkers. 2) Laboratory studies show caffeine improves insulin sensitivity. 3) My friend drinks coffee and doesn't have diabetes."
    )
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Full Source Code

For the complete implementation of the Multi-Provider Reasoning agent, refer to the [multi_provider_reasoning.py](https://github.com/Mathews-Tom/FastADK/blob/main/examples/advanced/multi_provider_reasoning.py) file in the examples directory.

## Key Takeaways

The Multi-Provider Reasoning example demonstrates several important patterns:

1. **Provider Agnosticism**: Building agents that can work with multiple LLM providers
2. **Dynamic Configuration**: Selecting providers based on available API keys
3. **Fallback Strategies**: Implementing robust error handling with provider fallbacks
4. **Provider-Specific Adaptations**: Adjusting prompts and parsing strategies for different providers
5. **Advanced Reasoning**: Implementing tools for complex critical thinking tasks

This approach allows your agents to be more resilient to API outages, take advantage of the best available models, and provide consistent functionality regardless of the underlying provider.

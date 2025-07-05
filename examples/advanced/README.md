# Advanced FastADK Examples

This directory contains advanced examples that demonstrate more complex FastADK features.

## Travel Assistant Example

The `travel_assistant.py` example demonstrates:

- Memory for conversation context
- Multiple tool implementations with caching, retries, and timeouts
- Error handling and fallback mechanisms
- API integrations with error handling
- Lifecycle hooks for tracking and monitoring

### Setup and Running

1. Install required dependencies:

   ```bash
   uv add httpx python-dotenv
   ```

2. Set up environment variables or a `.env` file:

   ```bash
   GEMINI_API_KEY=your_api_key_here
   
   # Optional for multi-provider support:
   OPENAI_API_KEY=your_api_key_here
   ANTHROPIC_API_KEY=your_api_key_here
   ```

3. Run the example:

   ```bash
   uv run travel_assistant.py
   ```

### Expected Output

```bash
=== Testing Travel Assistant with direct tool calls ===

Testing get_weather tool...
Weather in Tokyo: {'city': 'Tokyo', 'temperature': 24, 'condition': 'stormy', 'humidity': 84}

Testing get_attractions tool...
Attractions in Paris: {'city': 'Paris', 'attractions': [{'name': 'Eiffel Tower', 'rating': 3.9, 'category': 'Cultural', 'suggested_duration': '5 hours'}, {'name': 'Louvre Museum', 'rating': 4.9, 'category': 'Entertainment', 'suggested_duration': '5 hours'}, {'name': 'Notre-Dame Cathedral', 'rating': 4.9, 'category': 'Architecture', 'suggested_duration': '1 hours'}, {'name': 'Arc de Triomphe', 'rating': 4.0, 'category': 'Cultural', 'suggested_duration': '1 hours'}]}

Testing currency conversion...
Currency conversion: {'original': {'amount': 100, 'currency': 'USD'}, 'converted': {'amount': 85.0, 'currency': 'EUR'}, 'rate': 0.85, 'date': '2025-07-05'}

Testing itinerary generation...
Itinerary for New York: {'destination': 'New York', 'days': 2, 'itinerary': {'Day 1': {'morning': 'Times Square', 'afternoon': 'Empire State Building', 'evening': 'New York Gourmet Restaurant', 'note': 'Optional: Relaxation'}, 'Day 2': {'morning': 'Central Park', 'afternoon': 'Statue of Liberty', 'evening': 'Traditional New York Cuisine', 'note': 'Optional: Local tour'}}}

=== Testing complete agent with user queries ===

Test Query: I want to find a hotel in Rome for August 15-20
Response: Let me help you find hotels in Rome for August 15-20! I'll search for some options for you.

Based on your dates (August 15-20), I've found these hotels in Rome:

🏨 **Colosseum Luxury Suites** - ⭐4.7
   • Price: €189 per night
   • Amenities: WiFi, Pool, Restaurant, Bar, Airport Shuttle

🏨 **Vatican View Hotel** - ⭐4.5
   • Price: €162 per night
   • Amenities: WiFi, Spa, Restaurant, Gym

🏨 **Roman Empire Resort** - ⭐3.8
   • Price: €145 per night
   • Amenities: WiFi, Gym, Bar, Airport Shuttle

Would you like me to filter these options by budget (low, medium, high) or would you like information about attractions near any of these hotels?
```

## Workflow Demo Example

The `workflow_demo.py` example demonstrates:

- Sequential and parallel workflows
- Conditional branching in workflows
- Error handling and retries
- Data transformation and merging
- Composing workflow steps

### Setup and Running

1. Run the example:

   ```bash
   uv run workflow_demo.py
   ```

### Expected Output

```bash
🚀 FastADK Workflow Orchestration Demo
=======================================

🌟 Running Weather Workflow
========================
🔄 Loading data from weather...
🔍 Validating data...
⚠️ Validation error (will retry)...
🔍 Validating data...
✨ Enriching data...
🌤️ Analyzing weather data...

✅ Weather Workflow Result:
Execution time: 0.90s
Analysis: Great day to be outside!
========================


🌟 Running Finance Workflow
========================
🔄 Loading data from finance...
🔍 Validating data...
✨ Enriching data...
📈 Analyzing financial data...

✅ Finance Workflow Result:
Execution time: 0.90s
Analysis: Strong buy
========================


🌟 Running Parallel Analysis Workflow
==================================
🔄 Loading data from ['weather', 'finance']...
🔄 Loading data from ['weather', 'finance']...
🔍 Validating data...
⚠️ Validation error (will retry)...
🔍 Validating data...
⚠️ Validation error (will retry)...
🔍 Validating data...
✨ Enriching data...
🌤️ Analyzing weather data...
🔍 Validating data...
⚠️ Validation error (will retry)...
🔍 Validating data...
✨ Enriching data...
📈 Analyzing financial data...
📊 Formatting final results...

✅ Parallel Workflow Result:
Execution time: 1.20s
Insights: 2 found
  - Weather in New York: 72°F, sunny. Great day to be outside!
  - Stock AAPL: $178.72 (+1.25). Recommendation: Strong buy
==================================

🏁 All workflow demos completed!
```

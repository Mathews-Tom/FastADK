# Workflow Orchestration Demo

This example demonstrates FastADK's workflow orchestration capabilities, showing how to create, configure, and execute different types of workflows.

## Features Demonstrated

- Creating sequential and parallel workflows
- Using step decorators to define workflow steps
- Creating conditional branches in workflows
- Handling errors and retries in workflow steps
- Composing multiple workflow steps together
- Transforming and merging data between workflow steps

## Prerequisites

No external dependencies or API keys are required for this example. It runs entirely with simulated data.

To run the example:

```bash
uv run workflow_demo.py
```

## How It Works

This example creates three different workflow scenarios:

1. **Weather Workflow**: A sequential workflow that processes weather data through several steps
2. **Finance Workflow**: A workflow with conditional branching based on data quality
3. **Parallel Workflow**: A workflow that processes multiple data sources in parallel and merges the results

Each workflow demonstrates different capabilities of the FastADK workflow system:

### Sequential Workflow

The weather workflow demonstrates a simple sequential flow:

1. Load weather data from a simulated source
2. Validate the data structure with retry capabilities
3. Enrich the data with additional calculated fields
4. Analyze the weather data to produce recommendations

### Conditional Workflow

The finance workflow demonstrates conditional branching:

1. Load financial data from a simulated source
2. Validate the data with retry capabilities
3. Enrich the data with additional calculated fields
4. Check data quality with a conditional branch:
   - If quality is high: Perform detailed financial analysis
   - If quality is low: Return a basic response without analysis

### Parallel Workflow

The parallel workflow demonstrates processing multiple data sources simultaneously:

1. Load weather and financial data in parallel
2. Process each data source through its own pipeline
3. Merge the results from both pipelines into a single response

## Core Components

- **@step decorator**: Defines workflow steps with optional configuration like timeout and retry
- **@transform decorator**: Creates steps that transform data without async operations
- **@merge decorator**: Defines steps that combine results from multiple parallel processes
- **conditional()**: Creates branching logic based on data values
- **ParallelFlow**: Executes multiple steps simultaneously
- **SequentialFlow**: Executes steps one after another
- **Workflow**: Manages the execution of a workflow with telemetry and error handling

## Expected Output

When you run the script, you should see output similar to:

```bash
🚀 FastADK Workflow Orchestration Demo
=======================================

🌟 Running Weather Workflow
========================
🔄 Loading data from weather...
🔍 Validating data...
✨ Enriching data...
🌤️ Analyzing weather data...

✅ Weather Workflow Result:
Execution time: 1.23s
Analysis: Great day to be outside!
========================

🌟 Running Finance Workflow
========================
🔄 Loading data from finance...
🔍 Validating data...
✨ Enriching data...
📈 Analyzing financial data...

✅ Finance Workflow Result:
Execution time: 1.45s
Analysis: Strong buy
========================

🌟 Running Parallel Analysis Workflow
==================================
🔄 Loading data from weather...
🔄 Loading data from finance...
🔍 Validating data...
🔍 Validating data...
✨ Enriching data...
✨ Enriching data...
🌤️ Analyzing weather data...
📈 Analyzing financial data...
📊 Formatting final results...
Processing result: valid=True, has_analysis=True
Added weather insight: Weather in New York: 72°F, sunny. Great day to be outside!
Processing result: valid=True, has_analysis=True
Added finance insight: Stock AAPL: $178.72 (+1.25). Recommendation: Strong buy

✅ Parallel Workflow Result:
Execution time: 1.67s
Insights: 2 found
  - Weather in New York: 72°F, sunny. Great day to be outside!
  - Stock AAPL: $178.72 (+1.25). Recommendation: Strong buy
==================================

🏁 All workflow demos completed!
```

## Key Concepts

1. **Workflow Composition**: FastADK allows you to compose complex workflows from simple steps using operators like `>>` and classes like `SequentialFlow` and `ParallelFlow`.

2. **Error Handling**: Each step can be configured with retry logic, timeouts, and error handlers.

3. **Data Transformation**: Workflows can transform data between steps, with each step receiving the output of the previous step.

4. **Conditional Logic**: Workflows can include conditional branching based on data values or external conditions.

5. **Parallel Processing**: Multiple data streams can be processed simultaneously and later merged.

## Best Practices Demonstrated

- Breaking complex processes into discrete, reusable steps
- Using meaningful names for workflow steps
- Handling errors at the workflow step level
- Using conditional logic to branch workflows
- Transforming data between workflow steps
- Merging results from parallel processes

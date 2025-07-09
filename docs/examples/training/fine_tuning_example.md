# Fine-Tuning Example

This example demonstrates how to use FastADK's fine-tuning utilities to prepare data, create fine-tuning jobs, and use the resulting fine-tuned models.

## Features Demonstrated

- Converting training data between different format standards
- Creating and monitoring fine-tuning jobs
- Using fine-tuned models with FastADK
- Supporting multiple provider-specific formats
- Safe job configuration to minimize costs

## Prerequisites

To run this example with actual fine-tuning jobs:

```bash
# Install required packages
uv add openai

# Set your API key
export OPENAI_API_KEY=your_key_here
```

Note: Running this example with `run_actual_job=True` will create real fine-tuning jobs that may incur costs on your OpenAI account. The example is configured to use minimal data and the smallest model (gpt-3.5-turbo) with minimal epochs to keep costs low.

## How It Works

The example demonstrates two main capabilities:

### 1. Data Format Conversion

FastADK provides utilities to convert between different training data formats:

- **Alpaca Format**: A simple instruction-input-output format
- **OpenAI Format**: The format required for OpenAI fine-tuning (JSONL with messages)
- **Vertex Format**: The format used by Google's Vertex AI for tuning

The example:

1. Creates a sample dataset in Alpaca format
2. Converts it to OpenAI's format
3. Converts it to Vertex AI's format
4. Displays samples of the converted data

### 2. Fine-Tuning Job Management

The example shows how to:

1. Check available fine-tuning providers
2. Create a fine-tuning job configuration
3. Submit a job to the provider
4. Get job status information

For safety, the actual job creation is disabled by default (`run_actual_job=False`).

## Sample Data

The example uses a minimal dataset with just two examples:

1. A text summarization example
2. A creative writing example (haiku)

This small dataset is sufficient to demonstrate the conversion process but would be too small for actual effective fine-tuning in a production scenario.

## Expected Output

When you run the script, you should see output similar to:

```bash
=== Data Conversion Example ===
Created sample Alpaca data file: /tmp/tmpdirxyz123/alpaca_data.json

Converting from Alpaca to OpenAI format...

Converted OpenAI format data:
- Message exchange with 2 messages
- Message exchange with 2 messages

Converting from Alpaca to Vertex format...

Converted Vertex format data:
- Example: 'Summarize the following text. FastADK is a develo...' -> 'FastADK is a framework that simplifies AI agent dev...'
- Example: 'Write a haiku about programming. ...' -> 'Fingers on keyboard Logic flows through lines of co...'

=== Fine-Tuning Job Example ===
Available fine-tuning providers: ['openai', 'vertex']
Skipping actual job creation (run_actual_job=False)

To create an actual fine-tuning job:
1. Install the required packages: uv add openai
2. Set your API key: export OPENAI_API_KEY=your_key_here
3. Call this function with run_actual_job=True
```

If you set `run_actual_job=True` and have a valid API key, you'll see additional output about the created job:

```bash
Creating fine-tuning job with model: gpt-3.5-turbo
Created job with ID: ft-abc123xyz456
Job status: validating
Check the OpenAI dashboard for job progress

To check job status later, use:
job = await default_fine_tuner.get_job('ft-abc123xyz456', FineTuningProvider.OPENAI)
print(f'Job status: {job.status}')
```

## Key Concepts

1. **Data Format Conversion**: FastADK provides utilities to convert between different training data formats, making it easier to prepare data for different providers.

2. **Provider Abstraction**: The fine-tuning API abstracts away provider-specific details, allowing you to use a consistent interface across different LLM providers.

3. **Job Management**: FastADK includes utilities for creating, monitoring, and managing fine-tuning jobs.

4. **Cost Management**: The example demonstrates how to configure jobs to minimize costs during experimentation.

## Best Practices Demonstrated

- Converting data between formats to maximize dataset reuse
- Using minimal data and epochs for experimentation
- Providing safety guards against accidental job creation
- Checking provider availability before creating jobs
- Using temporary directories for training data files

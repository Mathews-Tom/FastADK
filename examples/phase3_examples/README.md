# FastADK Phase 3 Examples

This directory contains examples of FastADK's Phase 3 features, which include:

- Workflow orchestration
- Enhanced error handling and resilience
- Memory context management
- Streaming API features

## Workflow Demo

The `workflow_demo.py` file demonstrates FastADK's powerful workflow orchestration capabilities. It shows how to:

1. Create sequential workflows where steps execute in order
2. Create parallel workflows where steps execute concurrently
3. Use conditional branching for dynamic workflows
4. Merge results from multiple steps
5. Handle errors and retries automatically
6. Transform data between workflow steps

To run the workflow demo:

```bash
uv run examples/phase3_examples/workflow_demo.py
```

## Streaming API Demo

FastADK Phase 3 introduces streaming API features, including:

- Server-Sent Events (SSE) for real-time streaming of agent responses
- WebSocket support for bidirectional communication
- Progressive output streaming for LLM generations

To test streaming features, start the FastAPI server and access the streaming endpoints:

```bash
# Install the SSE library (optional but recommended)
uv add sse-starlette

# Run the server
uv run -m fastadk.cli.main run-server

# Access streaming endpoints at:
# - GET/POST /stream/agents/{agent_name}
# - WebSocket /ws/{client_id}
```

## Memory Context Management

Phase 3 enhances FastADK's memory capabilities with:

- Conversation context management
- Automatic context window sizing
- Long-term memory persistence
- Context summarization (preparing for future LLM-based summarization)

Memory context management is used internally by agents to maintain conversation state.
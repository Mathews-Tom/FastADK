# Streamlit Chat UI Example

## Overview

This example demonstrates how to create a simple yet effective chat interface for FastADK agents using Streamlit. The application provides a user-friendly web interface that lets users interact with any FastADK agent through a familiar chat UI.

## Features Demonstrated

- **Web UI Integration**: Creating a web interface for FastADK agents
- **Session State Management**: Maintaining conversation history between interactions
- **Asynchronous Processing**: Handling agent responses asynchronously
- **Token Usage Tracking**: Displaying token usage metrics in the UI
- **Dynamic Agent Loading**: Framework for loading different agent types (simplified in demo)
- **Chat History Display**: Rendering conversation history with proper formatting

## Running this Example

### 1. Install Dependencies

```bash
uv add streamlit
```

### 2. Run the Application

```bash
uv run -m streamlit run examples/ui/streamlit_chat_app.py
```

The application will launch in your default web browser. If it doesn't open automatically, navigate to the URL displayed in the terminal (typically <http://localhost:8501>).

## User Interface

The UI consists of:

- **Chat Window**: The main area showing the conversation history
- **Input Box**: A text input field at the bottom for user messages
- **Sidebar**: Contains agent information, token usage stats, and controls

### Sidebar Features

- **Agent Information**: Shows details about the currently loaded agent
- **Token Usage**: Displays current token consumption metrics
- **Clear Chat**: Button to reset the conversation
- **Settings**: Additional controls and information

## Implementation Details

The app is implemented with:

- Streamlit for the UI framework
- Asynchronous messaging with the FastADK agent
- Session state management for conversation persistence
- Markdown rendering for message formatting

## Customization

You can extend this example by:

- Loading agents dynamically from a dropdown menu
- Adding configuration options for the agent
- Implementing file upload for document-based conversations
- Adding visualization of agent reasoning process
- Creating a voice input/output option
- Implementing a chat export feature

## Integration with Your Own Agents

To use this UI with your own FastADK agents:

1. Modify the `load_agent_class()` function to load your custom agent
2. Add any necessary configuration options to the sidebar
3. Update token tracking if you've customized how tokens are counted
4. Add any special rendering for structured outputs your agent may provide

## Requirements

- fastadk
- streamlit
- An OpenAI API key (set as OPENAI_API_KEY environment variable) or appropriate API key for your chosen model provider.

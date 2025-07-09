"""
FastADK Streamlit Chat Application

This example demonstrates how to create a simple chat interface for FastADK agents using Streamlit.
To run this application:

1. Install dependencies:
    uv add streamlit

    # If you haven't installed FastADK yet:
    uv add fastadk

2. Run the app:
    uv run -m streamlit run streamlit_chat_app.py

The app connects to a FastADK agent and provides a chat interface for interaction.
It serves as a simple yet functional UI for your FastADK agents.

Note: Make sure you have the appropriate API keys set in your environment
variables before running the app (e.g., OPENAI_API_KEY).
"""

import asyncio
import os
import sys
from typing import Type

# Streamlit import is kept here, even though it might not be installed
# in the environment when checking the code. It's a runtime requirement.
import streamlit as st

# Add the project root to the Python path so we can import FastADK modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import FastADK components
from fastadk.core.agent import BaseAgent


# Use a simple demo agent if no specific agent is provided
class SimpleAgent(BaseAgent):
    """A simple agent for demo purposes."""

    _description = "A demo agent for the Streamlit chat interface"
    _model_name = "gpt-3.5-turbo"
    _provider = "openai"

    async def run(self, prompt: str) -> str:
        """Process the user's input and return a response."""
        return await super().run(prompt)


# Configure Streamlit page
st.set_page_config(
    page_title="FastADK Chat",
    page_icon="🚀",
    layout="centered",
    initial_sidebar_state="expanded",
)


def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_initialized" not in st.session_state:
        st.session_state.agent_initialized = False
    if "agent_instance" not in st.session_state:
        st.session_state.agent_instance = None
    if "token_usage" not in st.session_state:
        st.session_state.token_usage = {"prompt": 0, "completion": 0, "total": 0}


def load_agent_class() -> Type[BaseAgent]:
    """
    Load a FastADK agent class dynamically or use the default SimpleAgent.

    In a real application, you would implement logic to dynamically load
    agent classes from a module, similar to how the CLI does it.
    """
    # For this demo, we'll just return the SimpleAgent
    return SimpleAgent


def format_token_usage() -> str:
    """Format the token usage information for display."""
    usage = st.session_state.token_usage
    return (
        f"Prompt tokens: {usage['prompt']}, "
        f"Completion tokens: {usage['completion']}, "
        f"Total tokens: {usage['total']}"
    )


async def process_message(agent: BaseAgent, message: str) -> str:
    """Process a message through the agent."""
    try:
        response = await agent.run(message)

        # Update token usage if available
        # Note: We're accessing protected members for demo purposes
        if hasattr(agent, "_token_usage"):
            st.session_state.token_usage = {
                "prompt": agent._token_usage.prompt_tokens,
                "completion": agent._token_usage.completion_tokens,
                "total": agent._token_usage.total_tokens,
            }

        return response
    except Exception as e:
        # In a production app, you'd want more specific error handling
        return f"Error: {str(e)}"


def add_message(role: str, content: str):
    """Add a message to the chat history."""
    st.session_state.messages.append({"role": role, "content": content})


def handle_user_input():
    """Handle user input from the chat interface."""
    user_input = st.session_state.user_input
    add_message("user", user_input)

    # Clear the input
    st.session_state.user_input = ""

    # Get the agent instance
    agent = st.session_state.agent_instance

    # Create a placeholder for the assistant message
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")

        # Process the message asynchronously
        response = asyncio.run(process_message(agent, user_input))

        # Update the placeholder with the response
        message_placeholder.markdown(response)

    # Add the assistant's response to the chat history
    add_message("assistant", response)


def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()

    # Display header
    st.title("🚀 FastADK Chat")
    st.markdown(
        """
        Welcome to the FastADK Chat application!
        This simple interface allows you to interact with AI agents built using FastADK.
        """
    )

    # Sidebar with agent information and settings
    with st.sidebar:
        st.header("Agent Settings")

        # Load agent dynamically or use default
        agent_class = load_agent_class()

        # Display agent information
        st.subheader("Agent Information")
        st.markdown(f"**Name:** {agent_class.__name__}")
        st.markdown(f"**Description:** {getattr(agent_class, '_description', 'N/A')}")
        st.markdown(f"**Model:** {getattr(agent_class, '_model_name', 'N/A')}")
        st.markdown(f"**Provider:** {getattr(agent_class, '_provider', 'N/A')}")

        # Add clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.token_usage = {"prompt": 0, "completion": 0, "total": 0}
            st.rerun()

        # Show token usage information
        st.subheader("Token Usage")
        st.markdown(format_token_usage())

        # Add GitHub link
        st.markdown("---")
        st.markdown(
            "Built with [FastADK](https://github.com/Mathews-Tom/FastADK) - "
            "The Developer-Friendly Framework for AI Agents"
        )

    # Initialize agent if not already done
    if not st.session_state.agent_initialized:
        with st.spinner("Initializing agent..."):
            st.session_state.agent_instance = agent_class()
            st.session_state.agent_initialized = True

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    st.chat_input(
        placeholder="Type a message...",
        key="user_input",
        on_submit=handle_user_input,
    )


if __name__ == "__main__":
    main()

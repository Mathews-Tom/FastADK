"""
Context Policies Demo for FastADK.

This example demonstrates how to use context policies in FastADK to:
1. Manage context window size for LLM interactions
2. Implement different summarization strategies
3. Use token budget constraints
4. Apply dynamic context pruning
5. Prioritize specific types of context

Usage:
    1. Run the example:
        uv run examples/advanced/context_policies_demo.py
"""

import asyncio
import logging
import os
from typing import List

from dotenv import load_dotenv

from fastadk import Agent, BaseAgent, tool
from fastadk.core.context_policy import (
    ContextPolicy,
    MostRecentPolicy,
    SummarizeOlderPolicy,
)

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ConversationMessage:
    """Represents a message in a conversation."""

    def __init__(self, role: str, content: str, message_type: str = "general"):
        self.role = role
        self.content = content
        self.message_type = message_type  # can be "general", "important", "technical"
        self.id = str(hash(f"{role}:{content}:{message_type}"))

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "message_type": self.message_type,
        }


@Agent(
    model="gemini-1.5-pro",
    description="An agent that demonstrates context policy management",
    provider="gemini",  # Will fall back to simulated if no API key is available
    system_prompt="""
    You are a technical support agent that helps users with computer problems.
    You should respond to their queries based on the conversation history provided.
    """,
)
class ContextPolicyDemoAgent(BaseAgent):
    """Agent demonstrating different context policies."""

    def __init__(self) -> None:
        super().__init__()
        self.conversation_history: List[ConversationMessage] = []
        # Start with a default fixed window policy
        self.active_policy = "fixed_window"
        self._initialize_demo_conversation()

    def _initialize_demo_conversation(self) -> None:
        """Initialize a sample conversation for demonstration."""
        # This simulates a technical support conversation that grows in length
        demo_conversation = [
            (
                "user",
                "I'm having trouble with my laptop. It's running very slowly.",
                "important",
            ),
            (
                "agent",
                "I'm sorry to hear that. Can you tell me what operating system you're using?",
                "general",
            ),
            ("user", "I'm using Windows 10 Pro, version 21H2.", "technical"),
            ("agent", "Thank you. When did you first notice the slowdown?", "general"),
            (
                "user",
                "It started about a week ago after installing some updates.",
                "important",
            ),
            (
                "agent",
                "That could be related. Let's check a few things. How much RAM does your computer have?",
                "general",
            ),
            ("user", "I think it has 8GB of RAM.", "technical"),
            (
                "agent",
                "Let's check if any processes are using too much memory. Can you open Task Manager?",
                "general",
            ),
            (
                "user",
                "Yes, I have Task Manager open. I see Chrome is using a lot of memory.",
                "technical",
            ),
            (
                "agent",
                "How many Chrome tabs do you have open? Chrome can be memory-intensive with many tabs.",
                "general",
            ),
            (
                "user",
                "I have about 20 tabs open. Should I close some of them?",
                "general",
            ),
            (
                "agent",
                "Yes, that would help. Try closing tabs you're not actively using to free up memory.",
                "important",
            ),
            (
                "user",
                "I closed most of the tabs, but the system is still slow.",
                "general",
            ),
            (
                "agent",
                "Let's check for other issues. When was the last time you restarted your computer?",
                "general",
            ),
            ("user", "I haven't restarted it in about two weeks.", "general"),
            (
                "agent",
                "I recommend restarting your computer. This can clear temporary files and refresh system resources.",
                "important",
            ),
            ("user", "I'll restart it now. Give me a moment.", "general"),
            ("agent", "Take your time. I'll wait for your update.", "general"),
            (
                "user",
                "I restarted the computer and it seems a bit faster, but still not as fast as before.",
                "general",
            ),
            (
                "agent",
                "Let's check for malware. Do you have an antivirus program installed?",
                "general",
            ),
            ("user", "Yes, I have Windows Defender.", "technical"),
            (
                "agent",
                "Great. Let's run a full scan with Windows Defender to check for malware.",
                "general",
            ),
            (
                "user",
                "The scan is complete. It found and removed a few threats.",
                "important",
            ),
            (
                "agent",
                "That's good. Those threats might have been affecting performance. Is it running better now?",
                "general",
            ),
            ("user", "It's a bit better, but still not optimal.", "general"),
            (
                "agent",
                "Let's check disk usage. How much free space do you have on your main drive?",
                "general",
            ),
            ("user", "I have about 50GB free out of 500GB.", "technical"),
            (
                "agent",
                "That should be enough. Let's check if disk defragmentation might help. When was the last time you defragmented your drive?",
                "general",
            ),
            ("user", "I'm not sure if I've ever done that.", "general"),
            (
                "agent",
                "For Windows 10, you can use the Optimize Drives tool. Let me guide you through the process.",
                "technical",
            ),
        ]

        # Add the demo conversation to history
        for role, content, message_type in demo_conversation:
            self.conversation_history.append(
                ConversationMessage(
                    role=role, content=content, message_type=message_type
                )
            )

        logger.info(
            "Initialized conversation with %s messages", len(self.conversation_history)
        )

    def _get_context_policy(self) -> ContextPolicy:
        """Get the active context policy based on current settings."""
        if self.active_policy == "fixed_window":
            # Keep only the last 10 messages
            return MostRecentPolicy(max_messages=10)

        elif self.active_policy == "max_tokens":
            # Limit context to approximately 2000 tokens
            return SummarizeOlderPolicy(threshold_tokens=2000, max_recent_messages=10)

        elif self.active_policy == "budget_based":
            # For budget-based, we'll use MostRecentPolicy as a fallback
            # since the actual BudgetBasedContextPolicy isn't available
            return MostRecentPolicy(max_messages=15)

        elif self.active_policy == "summarization":
            # Summarize older parts of the conversation
            return SummarizeOlderPolicy(
                threshold_tokens=4000,
                max_recent_messages=5,
            )

        elif self.active_policy == "prioritization":
            # For prioritization, we'll use MostRecentPolicy as a fallback
            # since the actual PrioritizationContextPolicy isn't available
            return MostRecentPolicy(max_messages=12)

        # Default to fixed window if none specified
        return MostRecentPolicy(max_messages=10)

    @tool
    def add_message(
        self, role: str, content: str, message_type: str = "general"
    ) -> dict:
        """
        Add a new message to the conversation.

        Args:
            role: The speaker role ('user' or 'agent')
            content: The message content
            message_type: Type of message ('general', 'important', 'technical')

        Returns:
            Status of the operation
        """
        if role not in ["user", "agent"]:
            return {
                "success": False,
                "message": "Role must be 'user' or 'agent'",
            }

        if message_type not in ["general", "important", "technical"]:
            return {
                "success": False,
                "message": "Message type must be 'general', 'important', or 'technical'",
            }

        # Add the message to conversation history
        self.conversation_history.append(
            ConversationMessage(role=role, content=content, message_type=message_type)
        )

        logger.info("Added %s message from %s", message_type, role)

        return {
            "success": True,
            "message": "Message added to conversation",
            "conversation_length": len(self.conversation_history),
        }

    @tool
    def switch_context_policy(self, policy_type: str) -> dict:
        """
        Switch to a different context management policy.

        Args:
            policy_type: Type of policy ('fixed_window', 'max_tokens', 'budget_based',
                            'summarization', 'prioritization')

        Returns:
            Status of the operation
        """
        valid_policies = [
            "fixed_window",
            "max_tokens",
            "budget_based",
            "summarization",
            "prioritization",
        ]

        if policy_type not in valid_policies:
            return {
                "success": False,
                "message": f"Invalid policy type. Choose from: {', '.join(valid_policies)}",
            }

        self.active_policy = policy_type
        policy = self._get_context_policy()

        logger.info("Switched to %s context policy", policy_type)

        return {
            "success": True,
            "message": f"Switched to {policy_type} context policy",
            "policy_description": str(policy),
        }

    @tool
    async def get_context_snapshot(self) -> dict:
        """
        Get a snapshot of the current context based on active policy.

        Returns:
            The current context after policy application
        """
        # Apply the current policy to get context
        policy = self._get_context_policy()

        # Create messages list from conversation history
        messages = []
        for msg in self.conversation_history:
            messages.append(msg)

        # Apply the policy to get processed context
        processed_messages = await policy.apply(messages)

        # Extract messages for display
        filtered_messages = []
        for msg in processed_messages:
            filtered_messages.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                    "message_type": msg.message_type,
                }
            )

        # Calculate approximate tokens
        context_content = "\n".join([msg.content for msg in processed_messages])
        approximate_tokens = len(context_content.split()) * 1.3  # Rough estimate

        logger.info(
            "Generated context snapshot using %s policy: %s/%s messages",
            self.active_policy,
            len(filtered_messages),
            len(self.conversation_history),
        )

        return {
            "success": True,
            "policy_type": self.active_policy,
            "total_messages": len(self.conversation_history),
            "context_messages": len(filtered_messages),
            "approximate_tokens": int(approximate_tokens),
            "messages": filtered_messages,
        }

    @tool
    async def analyze_context_efficiency(self) -> dict:
        """
        Analyze and compare different context policies for the current conversation.

        Returns:
            Efficiency analysis of different policies
        """
        policies = [
            "fixed_window",
            "max_tokens",
            "budget_based",
            "summarization",
            "prioritization",
        ]

        results = []

        # Create messages list from conversation history
        messages = []
        for msg in self.conversation_history:
            messages.append(msg)

        # Apply each policy and analyze
        for policy_name in policies:
            # Save current policy
            current_policy = self.active_policy

            # Temporarily switch policy
            self.active_policy = policy_name
            policy = self._get_context_policy()

            # Apply the policy
            processed_messages = await policy.apply(messages)

            # Calculate metrics
            message_count = len(processed_messages)
            context_content = "\n".join([msg.content for msg in processed_messages])
            approximate_tokens = len(context_content.split()) * 1.3  # Rough estimate

            # Analyze message types retained
            message_types = {
                "important": 0,
                "technical": 0,
                "general": 0,
            }

            for msg in processed_messages:
                message_types[msg.message_type] += 1

            # Add results
            results.append(
                {
                    "policy": policy_name,
                    "messages_retained": message_count,
                    "retention_percentage": round(
                        (message_count / len(self.conversation_history)) * 100, 1
                    ),
                    "approximate_tokens": int(approximate_tokens),
                    "message_types_retained": message_types,
                }
            )

            # Restore original policy
            self.active_policy = current_policy

        logger.info("Analyzed efficiency of %s context policies", len(policies))

        return {
            "success": True,
            "total_conversation_messages": len(self.conversation_history),
            "policy_comparison": results,
        }


async def demonstrate_context_policies() -> None:
    """Run the context policies demonstration."""
    print("\n" + "=" * 60)
    print("🧠 FastADK Context Policies Demo")
    print("=" * 60)

    # Check if API key is available
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n⚠️  No GEMINI_API_KEY found in environment variables.")
        print("This demo will run with simulated responses.")
        print("For a better experience with real responses, set your API key:")
        print("  export GEMINI_API_KEY=your_api_key_here")

    # Initialize the agent
    agent = ContextPolicyDemoAgent()

    # Show initial conversation statistics
    print(f"\n📋 Loaded conversation with {len(agent.conversation_history)} messages")
    important_count = sum(
        1 for msg in agent.conversation_history if msg.message_type == "important"
    )
    technical_count = sum(
        1 for msg in agent.conversation_history if msg.message_type == "technical"
    )
    general_count = sum(
        1 for msg in agent.conversation_history if msg.message_type == "general"
    )

    print(f"  - Important messages: {important_count}")
    print(f"  - Technical messages: {technical_count}")
    print(f"  - General messages: {general_count}")

    # Demonstrate each context policy
    policies = [
        ("fixed_window", "Retains a fixed number of most recent messages"),
        ("max_tokens", "Limits context based on token count"),
        ("budget_based", "Allocates token budget to different parts of context"),
        ("summarization", "Summarizes older parts of the conversation"),
        ("prioritization", "Prioritizes messages based on importance"),
    ]

    for policy_name, description in policies:
        print(f"\n\n📌 {policy_name.upper()} POLICY")
        print(f"   {description}")
        print("-" * 60)

        # Switch to this policy
        result = await agent.execute_tool(
            "switch_context_policy", policy_type=policy_name
        )
        if result.get("success", False):
            print(f"✅ Switched to {policy_name} policy")
        else:
            print(
                f"❌ Failed to switch policy: {result.get('message', 'Unknown error')}"
            )
            continue

        # Get context snapshot with this policy
        snapshot = await (await agent.execute_tool("get_context_snapshot"))

        if snapshot.get("success", False):
            context_size = snapshot.get("context_messages")
            total_messages = snapshot.get("total_messages")
            retention_percentage = round((context_size / total_messages) * 100)

            print("\n📊 Policy Statistics:")
            print(
                f"  - Messages retained: {context_size}/{total_messages} ({retention_percentage}%)"
            )
            print(
                f"  - Approximate tokens: {snapshot.get('approximate_tokens', 'N/A')}"
            )

            print("\n📝 Retained Messages:")
            for i, msg in enumerate(snapshot.get("messages", []), 1):
                # Truncate long messages for display
                content = msg["content"]
                if len(content) > 50:
                    content = content[:47] + "..."

                # Add emoji based on message type
                if msg["message_type"] == "important":
                    emoji = "❗"
                elif msg["message_type"] == "technical":
                    emoji = "🔧"
                else:
                    emoji = "💬"

                print(f"  {i}. {emoji} {msg['role'].upper()}: {content}")
        else:
            print(
                f"❌ Failed to get context snapshot: {snapshot.get('message', 'Unknown error')}"
            )

    # Compare all policies
    print("\n\n📊 CONTEXT POLICY COMPARISON")
    print("-" * 60)

    analysis = await (await agent.execute_tool("analyze_context_efficiency"))

    if analysis.get("success", False):
        total_messages = analysis.get("total_conversation_messages")
        comparison = analysis.get("policy_comparison", [])

        print(f"Total conversation messages: {total_messages}")
        print("\nPolicy Efficiency Comparison:")

        # Create a table-like output
        print(
            "\n| Policy          | Messages | Retention % | Tokens | Important | Technical | General |"
        )
        print(
            "|-----------------|----------|-------------|--------|-----------|-----------|---------|"
        )

        for policy_data in comparison:
            policy_name = policy_data["policy"].ljust(15)
            messages = str(policy_data["messages_retained"]).ljust(8)
            retention = str(policy_data["retention_percentage"]) + "%"
            retention = retention.ljust(11)
            tokens = str(policy_data["approximate_tokens"]).ljust(6)

            message_types = policy_data["message_types_retained"]
            important = str(message_types["important"]).ljust(9)
            technical = str(message_types["technical"]).ljust(9)
            general = str(message_types["general"]).ljust(7)

            print(
                f"| {policy_name} | {messages} | {retention} | {tokens} | {important} | {technical} | {general} |"
            )

        # Provide recommendations
        print("\n🔍 Analysis & Recommendations:")
        print("  • Fixed Window: Simple but may lose important historical context")
        print(
            "  • Max Tokens: Efficient for LLM context limits but doesn't prioritize content"
        )
        print("  • Budget Based: Good balance between system info and history")
        print(
            "  • Summarization: Best for long conversations to retain key information"
        )
        print("  • Prioritization: Optimal when important messages must be preserved")

        print("\n💡 Recommended usage:")
        print("  - For short interactions: Fixed Window or Max Tokens")
        print("  - For complex support cases: Prioritization or Summarization")
        print("  - For balanced approach: Budget Based")
    else:
        print(
            f"❌ Failed to analyze context efficiency: {analysis.get('message', 'Unknown error')}"
        )

    # Add some additional messages to demonstrate context updating
    print("\n\n🔄 DEMONSTRATING DYNAMIC CONTEXT UPDATES")
    print("-" * 60)

    # Switch to summarization policy for this demonstration
    await agent.execute_tool("switch_context_policy", policy_type="summarization")

    print("Adding new messages to the conversation...")

    # Add some new messages
    new_messages = [
        (
            "user",
            "I've also noticed my battery is draining faster than before.",
            "important",
        ),
        (
            "agent",
            "Battery drain could be related to background processes. Let's check what's running on startup.",
            "general",
        ),
        ("user", "How do I check startup programs?", "general"),
        (
            "agent",
            "You can check startup programs in Task Manager. Go to the Startup tab.",
            "technical",
        ),
        ("user", "I see several programs set to start automatically.", "general"),
    ]

    for role, content, msg_type in new_messages:
        result = await agent.execute_tool(
            "add_message", role=role, content=content, message_type=msg_type
        )
        if result.get("success", False):
            print(f"  ➕ Added message from {role} (type: {msg_type})")
        else:
            print(
                f"  ❌ Failed to add message: {result.get('message', 'Unknown error')}"
            )

    # Show updated context
    print("\nUpdated context with summarization policy:")
    snapshot = await (await agent.execute_tool("get_context_snapshot"))

    if snapshot.get("success", False):
        context_size = snapshot.get("context_messages")
        total_messages = snapshot.get("total_messages")
        retention_percentage = round((context_size / total_messages) * 100)

        print("\n📊 Updated Statistics:")
        print(
            f"  - Messages retained: {context_size}/{total_messages} ({retention_percentage}%)"
        )
        print(f"  - Approximate tokens: {snapshot.get('approximate_tokens', 'N/A')}")

        # Look for summarized content
        has_summarization = False
        for msg in snapshot.get("messages", []):
            if "summary" in msg["content"].lower():
                has_summarization = True
                print("\n🔍 Context now includes summarized content!")
                break

        if not has_summarization:
            print("\n(Note: Add more messages to trigger summarization)")

    print("\n" + "=" * 60)
    print("🏁 FastADK - Context Policies Demo Completed")
    print("=" * 60 + "\n")


async def main() -> None:
    """Run the main demo."""
    await demonstrate_context_policies()


if __name__ == "__main__":
    asyncio.run(main())

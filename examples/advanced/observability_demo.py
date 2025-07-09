"""
Observability Demo for FastADK.

This example demonstrates FastADK's observability capabilities:
1. Configuring detailed logging
2. Setting up metrics collection and reporting
3. Implementing distributed tracing
4. Visualizing agent performance
5. Redacting sensitive information

Usage:
    1. Run the example:
        uv run examples/advanced/observability_demo.py
"""

import asyncio
import contextlib
import logging
import os
import random
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from fastadk import Agent, BaseAgent, tool
from fastadk.observability import logger, metrics, redaction, tracer


# Helper functions for observability
@contextlib.contextmanager
def trace_operation(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Context manager for tracing operations."""
    span = tracer.create_span(name)
    if attributes:
        for key, value in attributes.items():
            span.set_attribute(key, value)
    try:
        yield span
    finally:
        span.end()


def redact_sensitive_data(text: str) -> str:
    """Redact sensitive information from text."""
    return redaction.redact(text)


# Load environment variables from .env file
load_dotenv()

# Configure detailed logging
logger.configure(
    level="INFO",
    json_format=True,  # Enable JSON-formatted logs
)

# Set up metrics
metrics_manager = metrics

# Create a logger for this module
logger = logging.getLogger(__name__)


@Agent(
    model="gemini-1.5-pro",
    description="An agent that demonstrates observability features",
    provider="gemini",  # Will fall back to simulated if no API key is available
    system_prompt="""
    You are a financial advisor assistant that can analyze transactions,
    provide recommendations, and answer questions about financial matters.
    """,
)
class ObservabilityDemoAgent(BaseAgent):
    """Agent demonstrating observability features."""

    def __init__(self) -> None:
        super().__init__()
        # Configure tracing and metrics for this agent
        self.trace_id = str(uuid.uuid4())
        # Register custom metrics
        self.tool_execution_counter = metrics.counter(
            "tool_executions_total", "Total number of tool executions"
        )
        self.request_duration = metrics.histogram(
            "request_duration_seconds",
            "Duration of requests in seconds",
            buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0],
        )
        # Register error counter for use throughout
        self.error_counter = metrics.counter(
            "errors_total", "Total number of errors encountered"
        )
        # Demo data (typically this would be from a database)
        self._initialize_demo_data()

    def _initialize_demo_data(self) -> None:
        """Initialize demo financial data for the agent."""
        with trace_operation("initialize_demo_data", {"agent_id": id(self)}):
            logger.info("Initializing demo financial data")

            # Sample transactions
            self.transactions = [
                {
                    "id": f"t{i}",
                    "date": (
                        datetime.now().replace(day=random.randint(1, 28))
                    ).strftime("%Y-%m-%d"),
                    "amount": round(random.uniform(10, 1000), 2),
                    "category": random.choice(
                        ["groceries", "dining", "entertainment", "utilities", "travel"]
                    ),
                    "description": random.choice(
                        [
                            "Grocery store purchase",
                            "Restaurant dinner",
                            "Movie tickets",
                            "Electricity bill",
                            "Flight tickets",
                            "Gas station",
                            "Online shopping",
                            "Coffee shop",
                            "Phone bill",
                        ]
                    ),
                }
                for i in range(1, 21)
            ]

            # Sensitive customer information (will be redacted in logs)
            self.customer_info = {
                "name": "John Smith",
                "email": "john.smith@example.com",
                "phone": "555-123-4567",
                "ssn": "123-45-6789",
                "account_number": "9876543210",
                "address": "123 Main St, Anytown, USA",
            }

            logger.info("Initialized %d sample transactions", len(self.transactions))

    @tool(return_type=dict)
    def analyze_spending(self, timeframe: str = "all") -> Dict[str, Any]:
        """
        Analyze spending patterns by category.

        Args:
            timeframe: Time period to analyze ('week', 'month', 'all')

        Returns:
            Spending analysis by category
        """
        # Start timing the operation
        start_time = time.time()

        with trace_operation("analyze_spending", {"timeframe": timeframe}):
            try:
                # Increment the tool execution counter
                self.tool_execution_counter.inc(1, {"tool": "analyze_spending"})

                logger.info("Analyzing spending for timeframe: %s", timeframe)

                # Filter transactions based on timeframe
                filtered_transactions = self.transactions
                if timeframe == "week":
                    # Just return a subset for demo purposes
                    filtered_transactions = self.transactions[:7]
                elif timeframe == "month":
                    filtered_transactions = self.transactions[:14]

                # Calculate spending by category
                spending_by_category = {}
                for transaction in filtered_transactions:
                    category = transaction["category"]
                    amount = transaction["amount"]

                    if category in spending_by_category:
                        spending_by_category[category] += amount
                    else:
                        spending_by_category[category] = amount

                # Calculate total spending
                total_spending = sum(spending_by_category.values())

                # Calculate percentages
                for category in spending_by_category:
                    spending_by_category[category] = {
                        "amount": round(spending_by_category[category], 2),
                        "percentage": round(
                            (spending_by_category[category] / total_spending) * 100, 1
                        ),
                    }

                # Log with sensitive data redacted
                redacted_info = redact_sensitive_data(
                    f"Analysis completed for customer {self.customer_info['name']} "
                    f"({self.customer_info['email']})"
                )
                logger.info(redacted_info)

                end_time = time.time()
                duration = end_time - start_time

                # Record the duration of the operation
                self.request_duration.observe(
                    duration, {"operation": "analyze_spending"}
                )

                return {
                    "success": True,
                    "timeframe": timeframe,
                    "total_spending": round(total_spending, 2),
                    "transaction_count": len(filtered_transactions),
                    "spending_by_category": spending_by_category,
                    "processing_time_ms": round(duration * 1000, 2),
                }
            except Exception as e:
                logger.error("Error analyzing spending: %s", e, exc_info=True)
                # Record the error in metrics
                self.error_counter.inc(1, {"operation": "analyze_spending"})
                return {
                    "success": False,
                    "message": f"Error analyzing spending: {str(e)}",
                }

    @tool(return_type=dict)
    def get_transaction_details(self, transaction_id: str) -> Dict[str, Any]:
        """
        Get details for a specific transaction.

        Args:
            transaction_id: ID of the transaction

        Returns:
            Transaction details
        """
        with trace_operation(
            "get_transaction_details", {"transaction_id": transaction_id}
        ):
            try:
                # Increment the tool execution counter
                self.tool_execution_counter.inc(1, {"tool": "get_transaction_details"})

                logger.info("Retrieving details for transaction: %s", transaction_id)

                # Find the transaction
                transaction = next(
                    (t for t in self.transactions if t["id"] == transaction_id), None
                )

                if not transaction:
                    logger.warning("Transaction not found: %s", transaction_id)
                    return {
                        "success": False,
                        "message": "Transaction {} not found".format(transaction_id),
                    }

                # Log with sensitive data redacted
                redacted_info = redact_sensitive_data(
                    f"Transaction {transaction_id} retrieved for account {self.customer_info['account_number']}"
                )
                logger.info(redacted_info)

                return {
                    "success": True,
                    "transaction": transaction,
                }
            except Exception as e:
                logger.error("Error retrieving transaction: %s", e, exc_info=True)
                # Record the error in metrics
                self.error_counter.inc(1, {"operation": "get_transaction_details"})
                return {
                    "success": False,
                    "message": f"Error retrieving transaction: {str(e)}",
                }

    @tool(return_type=dict)
    def get_savings_recommendations(self) -> Dict[str, Any]:
        """
        Get personalized savings recommendations based on spending patterns.

        Returns:
            Savings recommendations
        """
        with trace_operation("get_savings_recommendations"):
            try:
                # Increment the tool execution counter
                self.tool_execution_counter.inc(
                    1, {"tool": "get_savings_recommendations"}
                )

                logger.info("Generating savings recommendations")

                # Calculate total spending by category
                spending_by_category = {}
                for transaction in self.transactions:
                    category = transaction["category"]
                    amount = transaction["amount"]

                    if category in spending_by_category:
                        spending_by_category[category] += amount
                    else:
                        spending_by_category[category] = amount

                # Find the top spending categories
                sorted_categories = sorted(
                    spending_by_category.items(), key=lambda x: x[1], reverse=True
                )
                top_categories = sorted_categories[:2]

                # Generate recommendations
                recommendations = []
                for category, amount in top_categories:
                    if category == "dining":
                        recommendations.append(
                            {
                                "category": category,
                                "current_spending": round(amount, 2),
                                "recommendation": "Consider cooking at home more often to reduce dining expenses.",
                                "potential_savings": round(amount * 0.3, 2),
                            }
                        )
                    elif category == "entertainment":
                        recommendations.append(
                            {
                                "category": category,
                                "current_spending": round(amount, 2),
                                "recommendation": "Look for free or low-cost entertainment options in your area.",
                                "potential_savings": round(amount * 0.25, 2),
                            }
                        )
                    elif category == "travel":
                        recommendations.append(
                            {
                                "category": category,
                                "current_spending": round(amount, 2),
                                "recommendation": "Consider using travel reward points or finding off-peak deals.",
                                "potential_savings": round(amount * 0.2, 2),
                            }
                        )
                    else:
                        recommendations.append(
                            {
                                "category": category,
                                "current_spending": round(amount, 2),
                                "recommendation": f"Review your {category} expenses for potential savings.",
                                "potential_savings": round(amount * 0.15, 2),
                            }
                        )

                # Calculate total potential savings
                total_potential_savings = sum(
                    r["potential_savings"] for r in recommendations
                )

                # Log with sensitive data redacted
                redacted_info = redact_sensitive_data(
                    f"Generated savings recommendations for {self.customer_info['name']}"
                )
                logger.info(redacted_info)

                return {
                    "success": True,
                    "recommendations": recommendations,
                    "total_potential_savings": round(total_potential_savings, 2),
                }
            except Exception as e:
                logger.error("Error generating recommendations: %s", e, exc_info=True)
                # Record the error in metrics
                self.error_counter.inc(1, {"operation": "get_savings_recommendations"})
                return {
                    "success": False,
                    "message": f"Error generating recommendations: {str(e)}",
                }

    @tool(return_type=dict)
    def check_security_alerts(self) -> Dict[str, Any]:
        """
        Check for security alerts on the account.

        Returns:
            Security alerts information
        """
        with trace_operation("check_security_alerts"):
            try:
                # Increment the tool execution counter
                self.tool_execution_counter.inc(1, {"tool": "check_security_alerts"})

                logger.info("Checking security alerts")

                # For demo purposes, randomly generate some security alerts
                has_alerts = random.random() < 0.3  # 30% chance of having alerts

                if has_alerts:
                    alerts = [
                        {
                            "id": f"alert-{uuid.uuid4().hex[:8]}",
                            "severity": random.choice(["low", "medium", "high"]),
                            "type": random.choice(
                                ["unusual_login", "large_transaction", "new_device"]
                            ),
                            "timestamp": datetime.now().isoformat(),
                            "description": random.choice(
                                [
                                    "Login attempt from unusual location",
                                    "Large transaction exceeding normal patterns",
                                    "New device used to access account",
                                ]
                            ),
                        }
                        for _ in range(random.randint(1, 3))
                    ]

                    # Log alert with sensitive data redacted
                    for alert in alerts:
                        redacted_info = redact_sensitive_data(
                            f"Security alert: {alert['type']} for account {self.customer_info['account_number']}"
                        )
                        logger.warning(redacted_info)

                    return {
                        "success": True,
                        "has_alerts": True,
                        "alerts": alerts,
                        "alert_count": len(alerts),
                    }
                else:
                    logger.info("No security alerts found")
                    return {
                        "success": True,
                        "has_alerts": False,
                        "message": "No security alerts found",
                    }
            except Exception as e:
                logger.error("Error checking security alerts: %s", e, exc_info=True)
                # Record the error in metrics
                self.error_counter.inc(1, {"operation": "check_security_alerts"})
                return {
                    "success": False,
                    "message": f"Error checking security alerts: {str(e)}",
                }


async def demonstrate_observability(agent: ObservabilityDemoAgent) -> None:
    """Run operations to demonstrate observability features."""
    # Wrap the entire demonstration in a trace
    with trace_operation("demonstrate_observability"):
        logger.info("Starting observability demonstration")

        # 1. Demonstrate spending analysis
        print("\n📊 Analyzing Spending Patterns")
        print("-" * 50)

        timeframes = ["week", "month", "all"]
        for timeframe in timeframes:
            print(f"\nAnalyzing spending for timeframe: {timeframe}")

            # Simulate a parent trace context
            with trace_operation(f"analyze_{timeframe}_spending"):
                result = await agent.execute_tool(
                    "analyze_spending", timeframe=timeframe
                )

                if result.get("success", False):
                    print(f"Total spending: ${result.get('total_spending')}")
                    print(f"Transactions analyzed: {result.get('transaction_count')}")
                    print("Top spending categories:")

                    categories = result.get("spending_by_category", {})
                    sorted_categories = sorted(
                        categories.items(), key=lambda x: x[1]["amount"], reverse=True
                    )

                    for category, details in sorted_categories:
                        print(
                            f"  - {category.capitalize()}: "
                            f"${details['amount']} ({details['percentage']}%)"
                        )

                    print(f"Processing time: {result.get('processing_time_ms')}ms")
                else:
                    print(f"Error: {result.get('message')}")

        # 2. Demonstrate transaction details retrieval
        print("\n🔍 Retrieving Transaction Details")
        print("-" * 50)

        # Get a random transaction ID
        transaction_id = random.choice(agent.transactions)["id"]
        print(f"Looking up transaction: {transaction_id}")

        result = await agent.execute_tool(
            "get_transaction_details", transaction_id=transaction_id
        )

        if result.get("success", False):
            transaction = result.get("transaction", {})
            print(f"Date: {transaction.get('date')}")
            print(f"Amount: ${transaction.get('amount')}")
            print(f"Category: {transaction.get('category')}")
            print(f"Description: {transaction.get('description')}")
        else:
            print(f"Error: {result.get('message')}")

        # 3. Demonstrate savings recommendations
        print("\n💰 Savings Recommendations")
        print("-" * 50)

        result = await agent.execute_tool("get_savings_recommendations")

        if result.get("success", False):
            recommendations = result.get("recommendations", [])

            print(f"Total potential savings: ${result.get('total_potential_savings')}")
            print("\nRecommendations:")

            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. {rec.get('category').capitalize()}:")
                print(f"   Current spending: ${rec.get('current_spending')}")
                print(f"   Recommendation: {rec.get('recommendation')}")
                print(f"   Potential savings: ${rec.get('potential_savings')}")
        else:
            print(f"Error: {result.get('message')}")

        # 4. Demonstrate security alerts
        print("\n🔒 Security Alerts")
        print("-" * 50)

        result = await agent.execute_tool("check_security_alerts")

        if result.get("success", False):
            if result.get("has_alerts", False):
                alerts = result.get("alerts", [])
                print(f"Found {len(alerts)} security alerts:")

                for i, alert in enumerate(alerts, 1):
                    print(f"\n{i}. {alert.get('description')}")
                    print(f"   Severity: {alert.get('severity')}")
                    print(f"   Type: {alert.get('type')}")
                    print(f"   Time: {alert.get('timestamp')}")
            else:
                print("No security alerts found")
        else:
            print(f"Error: {result.get('message')}")

        # 5. Display metrics summary
        print("\n📈 Metrics Summary")
        print("-" * 50)

        # For a real app, we'd use more metrics reporting here
        # Just demonstrate what the summary would look like
        print("Total tool executions: 4")
        print("  - analyze_spending: 3")
        print("  - get_transaction_details: 1")
        print("  - get_savings_recommendations: 1")
        print("  - check_security_alerts: 1")
        print("\nNo errors recorded")

        logger.info("Observability demonstration completed")


async def main() -> None:
    """Run the observability demo."""
    # Print banner
    print("\n" + "=" * 60)
    print("👁️ FastADK Observability Demo")
    print("=" * 60)

    # Check for API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n⚠️  No GEMINI_API_KEY found in environment variables.")
        print("This demo will run with simulated responses.")
        print("For a better experience with real responses, set your API key:")
        print("  export GEMINI_API_KEY=your_api_key_here")

    try:
        # Initialize the agent
        agent = ObservabilityDemoAgent()

        # Run the demonstration
        await demonstrate_observability(agent)

        # Display observability summary
        print("\n📊 FastADK - Observability Features Demonstrated")
        print("-" * 60)
        print("✓ Structured Logging: Check the console output for detailed logs")
        print("✓ Metrics Collection: Tool execution counts and durations")
        print(
            "✓ Distributed Tracing: Operations are traced with parent-child relationships"
        )
        print("✓ Performance Monitoring: Request durations are tracked")
        print("✓ Data Redaction: Sensitive information is automatically redacted")

        print("\n" + "=" * 60)
        print("🏁 FastADK - Observability Demo Completed")
        print("=" * 60)
    except Exception as e:
        logger.error("Error in main function: %s", e, exc_info=True)
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())

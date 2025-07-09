"""
Customer Support Agent Example

This example demonstrates how to create a customer support agent with FastADK.
The agent can handle product information requests, order status checks, refund requests,
and common technical support issues.

Usage:
    uv run examples/advanced/customer_support.py

Requirements:
    - fastadk
"""

import asyncio
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Any, List, Optional

from fastadk.core.agent import BaseAgent, tool
from fastadk.core.context_policy import ContextPolicy
from fastadk.core.exceptions import ToolError
from fastadk.memory.inmemory import InMemoryBackend
from fastadk.observability.logger import setup_logging

# Configure logging
setup_logging(level="INFO")
logger = logging.getLogger("customer_support")

# Mock database for customer support demo
PRODUCTS = {
    "PRD001": {
        "name": "SmartHome Hub",
        "price": 149.99,
        "category": "Smart Home",
        "in_stock": True,
        "description": "Central hub for controlling all your smart home devices",
        "warranty": "2 years limited warranty",
        "common_issues": [
            "Connection drops",
            "Device pairing problems",
            "App compatibility issues",
            "Firmware update failures",
        ],
    },
    "PRD002": {
        "name": "Wireless Earbuds Pro",
        "price": 99.99,
        "category": "Audio",
        "in_stock": True,
        "description": "Premium wireless earbuds with noise cancellation",
        "warranty": "1 year limited warranty",
        "common_issues": [
            "Bluetooth pairing issues",
            "One earbud not working",
            "Battery draining quickly",
            "Charging case not charging",
        ],
    },
    "PRD003": {
        "name": "UltraBook X1",
        "price": 1299.99,
        "category": "Computers",
        "in_stock": False,
        "description": "Lightweight, powerful laptop for professionals",
        "warranty": "3 years extended warranty available",
        "common_issues": [
            "Overheating during heavy use",
            "Battery life degradation",
            "Display flickering",
            "Wi-Fi connectivity problems",
        ],
    },
    "PRD004": {
        "name": '4K Smart TV 55"',
        "price": 599.99,
        "category": "TVs",
        "in_stock": True,
        "description": "Ultra HD smart TV with voice control",
        "warranty": "2 years limited warranty",
        "common_issues": [
            "Smart features not connecting to internet",
            "Remote control unresponsive",
            "Screen discoloration",
            "Apps crashing or freezing",
        ],
    },
    "PRD005": {
        "name": "Fitness Tracker Pro",
        "price": 79.99,
        "category": "Wearables",
        "in_stock": True,
        "description": "Advanced fitness tracking with heart rate monitoring",
        "warranty": "1 year limited warranty",
        "common_issues": [
            "Heart rate sensor inaccuracy",
            "Strap breaking or wearing out",
            "Battery life shorter than expected",
            "Syncing problems with smartphone app",
        ],
    },
}

ORDERS = {
    "ORD12345": {
        "customer_id": "CUST001",
        "date": "2025-06-25",
        "status": "Delivered",
        "tracking": "TRK789012345",
        "items": [
            {"product_id": "PRD001", "quantity": 1, "price": 149.99},
            {"product_id": "PRD005", "quantity": 1, "price": 79.99},
        ],
        "shipping_address": "123 Main St, Anytown, USA",
        "total": 229.98,
    },
    "ORD12346": {
        "customer_id": "CUST002",
        "date": "2025-06-27",
        "status": "Shipped",
        "tracking": "TRK789012346",
        "items": [{"product_id": "PRD004", "quantity": 1, "price": 599.99}],
        "shipping_address": "456 Oak Ave, Somewhere, USA",
        "total": 599.99,
    },
    "ORD12347": {
        "customer_id": "CUST003",
        "date": "2025-07-01",
        "status": "Processing",
        "tracking": None,
        "items": [
            {"product_id": "PRD002", "quantity": 2, "price": 199.98},
            {"product_id": "PRD005", "quantity": 1, "price": 79.99},
        ],
        "shipping_address": "789 Pine Rd, Elsewhere, USA",
        "total": 279.97,
    },
    "ORD12348": {
        "customer_id": "CUST001",
        "date": "2025-07-03",
        "status": "Processing",
        "tracking": None,
        "items": [{"product_id": "PRD003", "quantity": 1, "price": 1299.99}],
        "shipping_address": "123 Main St, Anytown, USA",
        "total": 1299.99,
    },
    "ORD12349": {
        "customer_id": "CUST004",
        "date": "2025-06-20",
        "status": "Cancelled",
        "tracking": None,
        "items": [{"product_id": "PRD002", "quantity": 1, "price": 99.99}],
        "shipping_address": "101 Maple St, Nowhere, USA",
        "total": 99.99,
    },
}

CUSTOMERS = {
    "CUST001": {
        "name": "John Smith",
        "email": "john.smith@example.com",
        "phone": "555-123-4567",
        "membership": "Premium",
        "since": "2022-03-15",
    },
    "CUST002": {
        "name": "Jane Doe",
        "email": "jane.doe@example.com",
        "phone": "555-234-5678",
        "membership": "Standard",
        "since": "2023-08-22",
    },
    "CUST003": {
        "name": "Robert Johnson",
        "email": "robert.johnson@example.com",
        "phone": "555-345-6789",
        "membership": "Premium",
        "since": "2021-11-05",
    },
    "CUST004": {
        "name": "Emily Williams",
        "email": "emily.williams@example.com",
        "phone": "555-456-7890",
        "membership": "Standard",
        "since": "2024-01-30",
    },
}

SUPPORT_TICKETS = {
    "TCKT001": {
        "customer_id": "CUST001",
        "product_id": "PRD001",
        "date_opened": "2025-06-29",
        "status": "Open",
        "issue": "Device not connecting to Wi-Fi",
        "priority": "Medium",
        "notes": ["Customer attempted factory reset without success"],
    },
    "TCKT002": {
        "customer_id": "CUST002",
        "product_id": "PRD004",
        "date_opened": "2025-06-26",
        "status": "Closed",
        "issue": "Remote control not working",
        "priority": "Low",
        "resolution": "Advised to replace batteries and re-pair remote",
        "notes": ["Issue resolved on first call"],
    },
    "TCKT003": {
        "customer_id": "CUST003",
        "product_id": "PRD002",
        "date_opened": "2025-07-02",
        "status": "Open",
        "issue": "Left earbud not charging",
        "priority": "High",
        "notes": ["Customer eligible for replacement under warranty"],
    },
}

# Knowledge base for common issues and solutions
KNOWLEDGE_BASE = {
    "smart_home_connectivity": {
        "issue": "Smart home device connectivity issues",
        "solutions": [
            "Ensure your Wi-Fi network is working correctly",
            "Check that the device is within range of your router",
            "Restart the device and your router",
            "Make sure your router firmware is up to date",
            "Verify that your mobile app is the latest version",
        ],
    },
    "bluetooth_pairing": {
        "issue": "Bluetooth pairing problems",
        "solutions": [
            "Turn Bluetooth off and on again on your device",
            "Make sure the product is in pairing mode (usually indicated by a flashing light)",
            "Clear previously paired devices from your Bluetooth settings",
            "Ensure the product has sufficient battery charge",
            "Reset the product to factory settings as a last resort",
        ],
    },
    "battery_issues": {
        "issue": "Battery draining quickly",
        "solutions": [
            "Check for background apps or processes that might be consuming power",
            "Lower screen brightness or adjust power settings",
            "Turn off features you're not using (like Bluetooth, Wi-Fi, or GPS)",
            "Update firmware to the latest version",
            "If persistent, the battery might need replacement",
        ],
    },
    "wifi_problems": {
        "issue": "Wi-Fi connectivity problems",
        "solutions": [
            "Restart your router and device",
            "Move closer to your router or consider a Wi-Fi extender",
            "Check if too many devices are connected to your network",
            "Try changing your Wi-Fi channel in router settings",
            "Update router and device firmware",
        ],
    },
    "software_crashes": {
        "issue": "App or software crashes",
        "solutions": [
            "Update to the latest software version",
            "Clear cache and temporary files",
            "Check for conflicting applications",
            "Try uninstalling and reinstalling the application",
            "Ensure your device meets the minimum system requirements",
        ],
    },
}


class PrioritySupportContextPolicy(ContextPolicy):
    """Custom context policy that prioritizes recent support ticket information."""

    def __init__(self, max_messages: int = 10, priority_keywords: List[str] = None):
        """
        Initialize the priority support context policy.

        Args:
            max_messages: Maximum number of messages to keep in context
            priority_keywords: Keywords to prioritize in context
        """
        self.max_messages = max_messages
        self.priority_keywords = priority_keywords or [
            "ticket",
            "issue",
            "problem",
            "error",
            "warranty",
            "refund",
        ]

    async def apply(self, history: List[Any]) -> List[Any]:
        """
        Apply the policy to the context.

        Args:
            history: The current conversation history

        Returns:
            Modified history with prioritized messages
        """
        if len(history) <= self.max_messages:
            return history

        # Always keep the system message
        system_message = next((m for m in history if m.get("role") == "system"), None)

        # Find messages containing priority keywords
        priority_messages = []
        for message in history:
            if message.get("role") == "system":
                continue

            content = message.get("content", "") or ""
            if any(keyword in content.lower() for keyword in self.priority_keywords):
                priority_messages.append(message)

        # Keep the most recent messages and priority messages
        remaining_slots = (
            self.max_messages - len(priority_messages) - (1 if system_message else 0)
        )
        recent_messages = history[-remaining_slots:] if remaining_slots > 0 else []

        # Combine and deduplicate messages while preserving order
        new_messages = []
        if system_message:
            new_messages.append(system_message)

        message_ids = set()
        for message in priority_messages + recent_messages:
            message_id = id(message)
            if message_id not in message_ids:
                new_messages.append(message)
                message_ids.add(message_id)

        # Sort by timestamp if available, otherwise keep original order
        return sorted(new_messages, key=lambda m: m.get("timestamp", id(m)))


class CustomerSupportAgent(BaseAgent):
    """Customer support agent that helps with product, order, and technical support inquiries."""

    _description = "Customer support agent for handling product inquiries, order status, and technical support"
    _model_name = "gpt-4"
    _provider = "openai"

    def __init__(self) -> None:
        """Initialize the customer support agent."""
        super().__init__()

        # Set custom system prompt
        self._system_prompt = """
        You are a helpful customer support agent for a consumer electronics company.
        You can provide assistance with product information, order status, refunds, and technical support.
        Always be polite, empathetic, and professional. Focus on solving the customer's problem efficiently.
        When handling technical issues, first check if there's a known solution in the knowledge base.
        For complex issues, guide the customer through troubleshooting steps or escalate to the appropriate department.
        For product questions, provide accurate information and suggest complementary products when appropriate.
        """

        # Configure memory and context policy
        self.memory_backend = InMemoryBackend()
        self.context_policies = [PrioritySupportContextPolicy(max_messages=8)]

        self.session_id = f"session_{random.randint(1000, 9999)}"
        self.customer_id = None

        logger.info(
            "Customer support agent initialized with session ID: %s", self.session_id
        )

    @tool
    async def get_product_info(self, product_id: str) -> str:
        """
        Get detailed information about a product.

        Args:
            product_id: The product ID to look up

        Returns:
            Detailed product information
        """
        product_id = product_id.upper()
        logger.info("Looking up product info for: %s", product_id)

        if product_id not in PRODUCTS:
            similar_products = []
            for pid, product in PRODUCTS.items():
                if (
                    product_id.lower() in pid.lower()
                    or product_id.lower() in product["name"].lower()
                ):
                    similar_products.append(pid)

            if similar_products:
                raise ToolError(
                    f"Product ID '{product_id}' not found. Did you mean one of these: {', '.join(similar_products)}?"
                )
            raise ToolError(f"Product ID '{product_id}' not found in our catalog.")

        product = PRODUCTS[product_id]
        stock_status = "In Stock" if product["in_stock"] else "Out of Stock"

        result = {
            "product_id": product_id,
            "name": product["name"],
            "price": f"${product['price']:.2f}",
            "category": product["category"],
            "description": product["description"],
            "warranty": product["warranty"],
            "availability": stock_status,
        }

        return json.dumps(result, indent=2)

    @tool
    async def search_products(self, query: str, category: Optional[str] = None) -> str:
        """
        Search for products by name or category.

        Args:
            query: Search terms to find products
            category: Optional category to filter results

        Returns:
            List of matching products
        """
        query = query.lower()
        results = []

        for product_id, product in PRODUCTS.items():
            if (
                query in product["name"].lower()
                or query in product["description"].lower()
            ) and (not category or category.lower() == product["category"].lower()):
                results.append(
                    {
                        "product_id": product_id,
                        "name": product["name"],
                        "price": f"${product['price']:.2f}",
                        "category": product["category"],
                        "availability": (
                            "In Stock" if product["in_stock"] else "Out of Stock"
                        ),
                    }
                )

        if not results:
            return json.dumps(
                {
                    "message": "No products found matching your search criteria.",
                    "results": [],
                }
            )

        return json.dumps({"results": results}, indent=2)

    @tool
    async def check_order_status(self, order_id: str) -> str:
        """
        Check the status of a customer order.

        Args:
            order_id: The order ID to look up

        Returns:
            Current order status and details
        """
        order_id = order_id.upper()
        logger.info("Checking order status for: %s", order_id)

        if order_id not in ORDERS:
            raise ToolError(f"Order ID '{order_id}' not found in our system.")

        order = ORDERS[order_id]

        # Get customer information
        customer = CUSTOMERS.get(order["customer_id"], {"name": "Customer"})

        # Format items with product names
        items = []
        for item in order["items"]:
            product = PRODUCTS.get(item["product_id"], {"name": "Unknown Product"})
            items.append(
                {
                    "product": product["name"],
                    "quantity": item["quantity"],
                    "price": f"${item['price']:.2f}",
                }
            )

        result = {
            "order_id": order_id,
            "customer_name": customer["name"],
            "date": order["date"],
            "status": order["status"],
            "tracking_number": order["tracking"] or "Not yet assigned",
            "items": items,
            "shipping_address": order["shipping_address"],
            "total": f"${order['total']:.2f}",
        }

        # Add estimated delivery date for shipped orders
        if order["status"] == "Shipped":
            ship_date = datetime.strptime(order["date"], "%Y-%m-%d")
            est_delivery = ship_date + timedelta(days=random.randint(3, 7))
            result["estimated_delivery"] = est_delivery.strftime("%Y-%m-%d")

        return json.dumps(result, indent=2)

    @tool
    async def get_customer_orders(self, customer_id: str) -> str:
        """
        Get all orders for a specific customer.

        Args:
            customer_id: The customer ID to look up

        Returns:
            List of customer orders
        """
        customer_id = customer_id.upper()

        if customer_id not in CUSTOMERS:
            raise ToolError(f"Customer ID '{customer_id}' not found in our system.")

        # Save the customer ID for the session
        self.customer_id = customer_id

        customer = CUSTOMERS[customer_id]
        customer_orders = []

        for order_id, order in ORDERS.items():
            if order["customer_id"] == customer_id:
                customer_orders.append(
                    {
                        "order_id": order_id,
                        "date": order["date"],
                        "status": order["status"],
                        "total": f"${order['total']:.2f}",
                        "items_count": sum(item["quantity"] for item in order["items"]),
                    }
                )

        result = {
            "customer_name": customer["name"],
            "email": customer["email"],
            "membership": customer["membership"],
            "orders_count": len(customer_orders),
            "orders": sorted(customer_orders, key=lambda x: x["date"], reverse=True),
        }

        return json.dumps(result, indent=2)

    @tool
    async def get_support_ticket(self, ticket_id: str) -> str:
        """
        Get details about a support ticket.

        Args:
            ticket_id: The ticket ID to look up

        Returns:
            Support ticket details
        """
        ticket_id = ticket_id.upper()

        if ticket_id not in SUPPORT_TICKETS:
            raise ToolError(f"Support ticket '{ticket_id}' not found in our system.")

        ticket = SUPPORT_TICKETS[ticket_id]
        customer = CUSTOMERS.get(ticket["customer_id"], {"name": "Customer"})
        product = PRODUCTS.get(ticket["product_id"], {"name": "Unknown Product"})

        result = {
            "ticket_id": ticket_id,
            "customer_name": customer["name"],
            "product": product["name"],
            "date_opened": ticket["date_opened"],
            "status": ticket["status"],
            "issue": ticket["issue"],
            "priority": ticket["priority"],
            "notes": ticket["notes"],
        }

        if "resolution" in ticket:
            result["resolution"] = ticket["resolution"]

        return json.dumps(result, indent=2)

    @tool
    async def create_support_ticket(
        self, customer_id: str, product_id: str, issue: str, priority: str = "Medium"
    ) -> str:
        """
        Create a new support ticket for a customer.

        Args:
            customer_id: The customer ID
            product_id: The product ID related to the issue
            issue: Description of the customer's issue
            priority: Ticket priority (Low, Medium, High)

        Returns:
            Confirmation with new ticket details
        """
        customer_id = customer_id.upper()
        product_id = product_id.upper()

        # Validate inputs
        if customer_id not in CUSTOMERS:
            raise ToolError(f"Customer ID '{customer_id}' not found in our system.")

        if product_id not in PRODUCTS:
            raise ToolError(f"Product ID '{product_id}' not found in our catalog.")

        priority = priority.capitalize()
        if priority not in ["Low", "Medium", "High"]:
            priority = "Medium"

        # Generate a new ticket ID
        ticket_id = f"TCKT{len(SUPPORT_TICKETS) + 1:03d}"

        # Create the ticket
        SUPPORT_TICKETS[ticket_id] = {
            "customer_id": customer_id,
            "product_id": product_id,
            "date_opened": datetime.now().strftime("%Y-%m-%d"),
            "status": "Open",
            "issue": issue,
            "priority": priority,
            "notes": ["Ticket created via support agent"],
        }

        customer = CUSTOMERS[customer_id]
        product = PRODUCTS[product_id]

        result = {
            "ticket_id": ticket_id,
            "message": "Support ticket created successfully",
            "customer_name": customer["name"],
            "product": product["name"],
            "issue": issue,
            "priority": priority,
            "status": "Open",
            "next_steps": "A support specialist will contact you within 24 hours.",
        }

        logger.info("Created support ticket %s for customer %s", ticket_id, customer_id)
        return json.dumps(result, indent=2)

    @tool
    async def search_knowledge_base(self, issue: str) -> str:
        """
        Search the knowledge base for solutions to common issues.

        Args:
            issue: Description of the issue or keywords

        Returns:
            Potential solutions from the knowledge base
        """
        issue = issue.lower()
        results = []

        # Check for product-specific issues
        for product_id, product in PRODUCTS.items():
            if "common_issues" in product:
                for common_issue in product["common_issues"]:
                    if any(
                        keyword in common_issue.lower() for keyword in issue.split()
                    ):
                        results.append(
                            {
                                "product": product["name"],
                                "issue": common_issue,
                                "recommendation": "Contact support for specific troubleshooting steps.",
                            }
                        )

        # Check the general knowledge base
        for kb_id, kb_item in KNOWLEDGE_BASE.items():
            relevance_score = 0
            for keyword in issue.split():
                if keyword in kb_id or keyword in kb_item["issue"].lower():
                    relevance_score += 1

            if relevance_score > 0:
                results.append(
                    {
                        "issue": kb_item["issue"],
                        "solutions": kb_item["solutions"],
                        "relevance": relevance_score,
                    }
                )

        # Sort by relevance
        results.sort(key=lambda x: x.get("relevance", 0), reverse=True)

        if not results:
            return json.dumps(
                {
                    "message": "No specific solutions found in our knowledge base.",
                    "general_advice": "Please provide more details about your issue or contact our support team.",
                }
            )

        return json.dumps(
            {"results": results[:3]}, indent=2
        )  # Return top 3 most relevant results

    @tool
    async def process_refund_request(
        self, order_id: str, reason: str, full_refund: bool = True
    ) -> str:
        """
        Process a refund request for an order.

        Args:
            order_id: The order ID to refund
            reason: Reason for the refund request
            full_refund: Whether to process a full or partial refund

        Returns:
            Refund request confirmation
        """
        order_id = order_id.upper()

        if order_id not in ORDERS:
            raise ToolError(f"Order ID '{order_id}' not found in our system.")

        order = ORDERS[order_id]

        # Check if order is eligible for refund
        if order["status"] == "Cancelled":
            raise ToolError("This order has already been cancelled and refunded.")

        # Check refund policy based on order date
        order_date = datetime.strptime(order["date"], "%Y-%m-%d")
        current_date = datetime.now()
        days_since_order = (current_date - order_date).days

        refund_policy = {
            "within_30_days": "Full refund available",
            "30_to_60_days": "Store credit or exchange only",
            "beyond_60_days": "Outside standard return window, special approval required",
        }

        refund_status = (
            "within_30_days"
            if days_since_order <= 30
            else "30_to_60_days" if days_since_order <= 60 else "beyond_60_days"
        )

        # Calculate refund amount
        refund_amount = order["total"] if full_refund else order["total"] * 0.85

        result = {
            "order_id": order_id,
            "refund_status": "Processing",
            "refund_amount": f"${refund_amount:.2f}",
            "reason": reason,
            "policy_applied": refund_policy[refund_status],
            "estimated_processing_time": "3-5 business days",
            "next_steps": "You will receive an email confirmation once the refund is processed.",
        }

        logger.info("Processed refund request for order %s", order_id)
        return json.dumps(result, indent=2)

    @tool
    async def check_warranty(self, product_id: str, purchase_date: str) -> str:
        """
        Check warranty status for a product.

        Args:
            product_id: The product ID to check
            purchase_date: Date of purchase in YYYY-MM-DD format

        Returns:
            Warranty status and details
        """
        product_id = product_id.upper()

        if product_id not in PRODUCTS:
            raise ToolError(f"Product ID '{product_id}' not found in our catalog.")

        product = PRODUCTS[product_id]

        try:
            purchase_date = datetime.strptime(purchase_date, "%Y-%m-%d")
        except ValueError:
            raise ToolError("Invalid date format. Please use YYYY-MM-DD format.")

        # Parse warranty period from the product data
        warranty_text = product["warranty"]
        warranty_years = 1  # Default

        if "year" in warranty_text:
            try:
                warranty_years = int(warranty_text.split()[0])
            except (ValueError, IndexError):
                pass

        warranty_end_date = purchase_date + timedelta(days=365 * warranty_years)
        current_date = datetime.now()

        is_under_warranty = current_date <= warranty_end_date
        days_remaining = (
            (warranty_end_date - current_date).days if is_under_warranty else 0
        )

        result = {
            "product": product["name"],
            "warranty_coverage": product["warranty"],
            "purchase_date": purchase_date.strftime("%Y-%m-%d"),
            "warranty_end_date": warranty_end_date.strftime("%Y-%m-%d"),
            "status": "Active" if is_under_warranty else "Expired",
            "days_remaining": max(0, days_remaining),
        }

        if not is_under_warranty:
            result["options"] = [
                "Extended warranty purchase may be available",
                "Check with customer service for repair options",
                "Consider upgrade options for older products",
            ]

        return json.dumps(result, indent=2)


async def main() -> None:
    """Run an interactive demo of the customer support agent."""
    print("\n" + "=" * 60)
    print("  🎧  CUSTOMER SUPPORT AGENT DEMO  💬")
    print("=" * 60)
    print("\nWelcome to the Customer Support Agent demo!")
    print("Ask about products, check order status, or get technical support.")
    print("\nExample queries:")
    print("- Tell me about your SmartHome Hub")
    print("- I'm having trouble with my Wireless Earbuds")
    print("- What's the status of my order ORD12345?")
    print("- I need help with Bluetooth pairing issues")
    print("- I want to return my order and get a refund")
    print("\nType 'exit' to end the demo.")
    print("=" * 60)

    agent = CustomerSupportAgent()

    while True:
        user_input = input("\n💬 You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nThank you for using our customer support service. Goodbye!")
            break

        print("\n⏳ Processing your request...")
        try:
            response = await agent.run(user_input)
            print(f"\n🎧 Support Agent: {response}")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            logger.exception("Error during agent execution")


if __name__ == "__main__":
    asyncio.run(main())

"""
Financial Assistant Agent Example

This example demonstrates how to create a financial assistant agent with FastADK.
The agent can provide financial advice, look up stock information, and calculate
simple financial metrics.

Usage:
    uv run examples/advanced/finance_assistant.py

Requirements:
    - fastadi
    - aiohttp (for stock data fetching)
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Union

from fastadk.core.agent import BaseAgent, tool
from fastadk.core.exceptions import ToolError
from fastadk.memory.inmemory import InMemoryBackend
from fastadk.tokens.models import TokenBudget


class FinancialAssistant(BaseAgent):
    """Financial assistant agent that provides financial advice and information."""

    _description = (
        "Financial advisor that helps with personal finance and investment decisions"
    )
    _model_name = "gpt-4"  # Use a capable model for financial reasoning
    _provider = "openai"

    def __init__(self) -> None:
        """Initialize the financial assistant agent."""
        super().__init__()

        # Configure memory
        self.memory_backend = InMemoryBackend()

        # Set token budget
        self.token_budget = TokenBudget(
            max_tokens_per_session=20000,
            max_tokens_per_request=4000,
            action_on_exceed="warn",
        )

        # Load financial constants
        self._tax_brackets = {
            "single": [
                {"max": 11000, "rate": 0.10},
                {"max": 44725, "rate": 0.12},
                {"max": 95375, "rate": 0.22},
                {"max": 182100, "rate": 0.24},
                {"max": 231250, "rate": 0.32},
                {"max": 578125, "rate": 0.35},
                {"max": float("inf"), "rate": 0.37},
            ],
            "married": [
                {"max": 22000, "rate": 0.10},
                {"max": 89450, "rate": 0.12},
                {"max": 190750, "rate": 0.22},
                {"max": 364200, "rate": 0.24},
                {"max": 462500, "rate": 0.32},
                {"max": 693750, "rate": 0.35},
                {"max": float("inf"), "rate": 0.37},
            ],
        }

        # Mock stock data (in a real implementation, this would come from an API)
        self._mock_stocks = {
            "AAPL": {"name": "Apple Inc.", "price": 185.92, "change": 0.75},
            "MSFT": {"name": "Microsoft Corporation", "price": 399.04, "change": -1.23},
            "GOOGL": {"name": "Alphabet Inc.", "price": 147.60, "change": 0.42},
            "AMZN": {"name": "Amazon.com, Inc.", "price": 178.22, "change": 1.05},
            "META": {"name": "Meta Platforms, Inc.", "price": 434.99, "change": 2.58},
            "TSLA": {"name": "Tesla, Inc.", "price": 175.60, "change": -3.40},
            "NVDA": {"name": "NVIDIA Corporation", "price": 116.00, "change": 0.95},
        }

    @tool
    async def get_stock_price(self, symbol: str) -> str:
        """
        Get the current stock price for a company.

        Args:
            symbol: Stock ticker symbol (e.g., AAPL, MSFT)

        Returns:
            Current stock price information

        Raises:
            ToolError: If the stock symbol is not found
        """
        symbol = symbol.upper()

        # In a real implementation, you would fetch from a stock API
        # For demo purposes, we're using mock data
        if symbol not in self._mock_stocks:
            raise ToolError(f"Stock symbol '{symbol}' not found")

        stock = self._mock_stocks[symbol]
        change_symbol = "+" if stock["change"] > 0 else ""

        return (
            f"{stock['name']} ({symbol}): ${stock['price']:.2f} "
            f"({change_symbol}{stock['change']:.2f}%)"
        )

    @tool
    async def calculate_compound_interest(
        self,
        principal: float,
        rate: float,
        time_years: int,
        compounds_per_year: int = 12,
    ) -> str:
        """
        Calculate compound interest for an investment.

        Args:
            principal: Initial investment amount
            rate: Annual interest rate (as a decimal, e.g., 0.05 for 5%)
            time_years: Number of years to invest
            compounds_per_year: Number of times interest compounds per year (default: 12, monthly)

        Returns:
            Compound interest calculation results
        """
        if principal <= 0:
            raise ToolError("Principal must be greater than zero")
        if rate < 0:
            raise ToolError("Interest rate cannot be negative")
        if time_years <= 0:
            raise ToolError("Time must be greater than zero")
        if compounds_per_year <= 0:
            raise ToolError("Compounds per year must be greater than zero")

        # Calculate compound interest
        n = compounds_per_year
        t = time_years
        r = rate
        p = principal

        # Compound interest formula: A = P(1 + r/n)^(nt)
        amount = p * (1 + r / n) ** (n * t)
        interest_earned = amount - p

        yearly_breakdown = []
        current_amount = p

        for year in range(1, t + 1):
            current_amount = p * (1 + r / n) ** (n * year)
            yearly_breakdown.append(
                {
                    "year": year,
                    "amount": round(current_amount, 2),
                    "interest_to_date": round(current_amount - p, 2),
                }
            )

        result = {
            "initial_investment": p,
            "annual_rate": f"{r:.2%}",
            "compound_frequency": f"{n} times per year",
            "time_period": f"{t} years",
            "final_amount": round(amount, 2),
            "interest_earned": round(interest_earned, 2),
            "yearly_breakdown": yearly_breakdown[:5],  # Show first 5 years
        }

        if t > 5:
            result["note"] = f"Showing first 5 years of a {t}-year investment."

        return json.dumps(result, indent=2)

    @tool
    async def calculate_mortgage_payment(
        self, loan_amount: float, annual_interest_rate: float, loan_term_years: int
    ) -> str:
        """
        Calculate monthly mortgage payment.

        Args:
            loan_amount: Total loan amount in dollars
            annual_interest_rate: Annual interest rate (as a decimal, e.g., 0.05 for 5%)
            loan_term_years: Loan term in years

        Returns:
            Monthly payment and loan amortization summary
        """
        if loan_amount <= 0:
            raise ToolError("Loan amount must be greater than zero")
        if annual_interest_rate < 0:
            raise ToolError("Interest rate cannot be negative")
        if loan_term_years <= 0:
            raise ToolError("Loan term must be greater than zero")

        # Calculate monthly payment
        p = loan_amount
        r = annual_interest_rate / 12  # Monthly interest rate
        n = loan_term_years * 12  # Total number of payments

        # Monthly payment formula: M = P[r(1+r)^n]/[(1+r)^n-1]
        if r == 0:
            # Handle special case of zero interest
            monthly_payment = p / n
        else:
            monthly_payment = p * (r * (1 + r) ** n) / ((1 + r) ** n - 1)

        total_payment = monthly_payment * n
        total_interest = total_payment - p

        # Calculate yearly breakdown for first few years
        remaining_balance = p
        yearly_breakdown = []

        for year in range(1, min(6, loan_term_years + 1)):
            year_interest = 0
            year_principal = 0

            for month in range(1, 13):
                if year * 12 + month <= n:
                    month_interest = remaining_balance * r
                    month_principal = monthly_payment - month_interest

                    remaining_balance -= month_principal
                    year_interest += month_interest
                    year_principal += month_principal

            yearly_breakdown.append(
                {
                    "year": year,
                    "principal_paid": round(year_principal, 2),
                    "interest_paid": round(year_interest, 2),
                    "remaining_balance": round(remaining_balance, 2),
                }
            )

        result = {
            "loan_amount": p,
            "annual_interest_rate": f"{annual_interest_rate:.2%}",
            "loan_term": f"{loan_term_years} years",
            "monthly_payment": round(monthly_payment, 2),
            "total_payment": round(total_payment, 2),
            "total_interest": round(total_interest, 2),
            "yearly_breakdown": yearly_breakdown,
        }

        if loan_term_years > 5:
            result["note"] = (
                f"Showing first 5 years of a {loan_term_years}-year mortgage."
            )

        return json.dumps(result, indent=2)

    @tool
    async def estimate_tax(self, income: float, filing_status: str = "single") -> str:
        """
        Estimate income tax based on filing status and income.

        Args:
            income: Annual income in dollars
            filing_status: Tax filing status ("single" or "married")

        Returns:
            Estimated tax breakdown
        """
        if income < 0:
            raise ToolError("Income cannot be negative")

        filing_status = filing_status.lower()
        if filing_status not in ["single", "married"]:
            raise ToolError('Filing status must be "single" or "married"')

        brackets = self._tax_brackets[filing_status]
        tax = 0
        previous_max = 0
        tax_breakdown = []

        for bracket in brackets:
            max_amount = bracket["max"]
            rate = bracket["rate"]

            if income > previous_max:
                taxable_in_bracket = min(income, max_amount) - previous_max
                tax_in_bracket = taxable_in_bracket * rate
                tax += tax_in_bracket

                tax_breakdown.append(
                    {
                        "bracket": (
                            f"${previous_max:,.0f} to ${max_amount:,.0f}"
                            if max_amount != float("inf")
                            else f"Over ${previous_max:,.0f}"
                        ),
                        "rate": f"{rate:.1%}",
                        "taxable_amount": round(taxable_in_bracket, 2),
                        "tax": round(tax_in_bracket, 2),
                    }
                )

            previous_max = max_amount

            if income <= max_amount:
                break

        # Calculate effective tax rate
        effective_rate = tax / income if income > 0 else 0

        result = {
            "filing_status": filing_status,
            "annual_income": round(income, 2),
            "total_tax": round(tax, 2),
            "effective_tax_rate": f"{effective_rate:.2%}",
            "marginal_tax_rate": f"{brackets[len(tax_breakdown) - 1]['rate']:.1%}",
            "tax_breakdown": tax_breakdown,
        }

        return json.dumps(result, indent=2)

    @tool
    async def calculate_retirement_savings(
        self,
        current_age: int,
        retirement_age: int,
        current_savings: float,
        monthly_contribution: float,
        expected_return: float,
        inflation_rate: float = 0.025,
    ) -> str:
        """
        Calculate retirement savings projection.

        Args:
            current_age: Current age in years
            retirement_age: Target retirement age in years
            current_savings: Current retirement savings in dollars
            monthly_contribution: Monthly contribution to retirement account
            expected_return: Expected annual return rate (as a decimal, e.g., 0.07 for 7%)
            inflation_rate: Expected annual inflation rate (default: 0.025 for 2.5%)

        Returns:
            Retirement savings projection
        """
        if current_age >= retirement_age:
            raise ToolError("Retirement age must be greater than current age")
        if current_savings < 0:
            raise ToolError("Current savings cannot be negative")
        if monthly_contribution < 0:
            raise ToolError("Monthly contribution cannot be negative")
        if expected_return < 0:
            raise ToolError("Expected return cannot be negative")
        if inflation_rate < 0:
            raise ToolError("Inflation rate cannot be negative")

        years_to_retirement = retirement_age - current_age
        real_return = (1 + expected_return) / (1 + inflation_rate) - 1

        # Calculate future value of current savings
        future_savings = current_savings * (1 + real_return) ** years_to_retirement

        # Calculate future value of monthly contributions
        monthly_real_return = real_return / 12
        months = years_to_retirement * 12

        if monthly_real_return == 0:
            future_contributions = monthly_contribution * months
        else:
            future_contributions = (
                monthly_contribution
                * ((1 + monthly_real_return) ** months - 1)
                / monthly_real_return
            )

        total_savings = future_savings + future_contributions

        # Calculate withdrawal phase (4% rule)
        annual_withdrawal = total_savings * 0.04
        monthly_withdrawal = annual_withdrawal / 12

        # Projection by decade
        projection = []
        savings = current_savings
        annual_contribution = monthly_contribution * 12

        for year in range(current_age, retirement_age + 1, 5):
            if year <= current_age:
                continue

            years_passed = year - current_age
            savings = current_savings * (1 + real_return) ** years_passed

            if monthly_real_return == 0:
                contributions_value = annual_contribution * years_passed
            else:
                contributions_value = (
                    monthly_contribution
                    * ((1 + monthly_real_return) ** (years_passed * 12) - 1)
                    / monthly_real_return
                )

            savings += contributions_value

            projection.append(
                {
                    "age": year,
                    "years_invested": years_passed,
                    "savings_in_today_dollars": round(savings, 2),
                }
            )

        result = {
            "current_age": current_age,
            "retirement_age": retirement_age,
            "years_to_retirement": years_to_retirement,
            "current_savings": round(current_savings, 2),
            "monthly_contribution": round(monthly_contribution, 2),
            "expected_return": f"{expected_return:.2%}",
            "inflation_adjusted_return": f"{real_return:.2%}",
            "projected_savings_at_retirement": round(total_savings, 2),
            "estimated_monthly_withdrawal": round(monthly_withdrawal, 2),
            "estimated_annual_withdrawal": round(annual_withdrawal, 2),
            "projection_by_age": projection,
        }

        return json.dumps(result, indent=2)


async def main() -> None:
    """Run the financial assistant in interactive mode."""
    print("🤖 Financial Assistant - Type 'exit' to quit")
    print("----------------------------------------------")
    print("Ask me about: stock prices, compound interest calculations,")
    print("mortgage payments, tax estimates, or retirement planning.")
    print("----------------------------------------------")

    agent = FinancialAssistant()

    while True:
        user_input = input("\n💬 You: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye! 👋")
            break

        print("\n⏳ Thinking...")
        response = await agent.run(user_input)
        print(f"\n🤖 Assistant: {response}")


if __name__ == "__main__":
    asyncio.run(main())

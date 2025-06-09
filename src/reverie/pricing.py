"""
Pricing module for calculating Claude conversation costs.

Fetches model pricing data from LiteLLM and calculates costs based on token usage.
"""

from enum import Enum

import httpx
from pydantic import BaseModel, Field


class CostMode(str, Enum):
    """Cost calculation mode."""

    AUTO = "auto"  # Use costUSD if available, otherwise calculate
    CALCULATE = "calculate"  # Always calculate from tokens
    DISPLAY = "display"  # Always use pre-calculated costUSD


class ModelPricing(BaseModel):
    """Pricing information for a specific model."""

    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None
    cache_creation_input_token_cost: float | None = None
    cache_read_input_token_cost: float | None = None


class TokenUsage(BaseModel):
    """Token usage for a single interaction."""

    input_tokens: int = Field(default=0, alias="input_tokens")
    output_tokens: int = Field(default=0, alias="output_tokens")
    cache_creation_input_tokens: int | None = Field(
        default=None, alias="cache_creation_input_tokens"
    )
    cache_read_input_tokens: int | None = Field(default=None, alias="cache_read_input_tokens")


class PricingFetcher:
    """Fetches and caches model pricing data from LiteLLM."""

    LITELLM_PRICING_URL = (
        "https://raw.githubusercontent.com/BerriAI/litellm/main/"
        "model_prices_and_context_window.json"
    )

    def __init__(self) -> None:
        self._cached_pricing: dict[str, ModelPricing] | None = None

    async def fetch_model_pricing(self) -> dict[str, ModelPricing]:
        """Fetch model pricing data from LiteLLM repository."""
        if self._cached_pricing is not None:
            return self._cached_pricing

        async with httpx.AsyncClient() as client:
            response = await client.get(self.LITELLM_PRICING_URL)
            response.raise_for_status()

            data = response.json()
            pricing: dict[str, ModelPricing] = {}

            for model_name, model_data in data.items():
                if isinstance(model_data, dict):
                    try:
                        pricing[model_name] = ModelPricing(**model_data)
                    except Exception:
                        continue

            self._cached_pricing = pricing
            return pricing

    def fetch_model_pricing_sync(self) -> dict[str, ModelPricing]:
        """Synchronous version of fetch_model_pricing."""
        if self._cached_pricing is not None:
            return self._cached_pricing

        with httpx.Client() as client:
            response = client.get(self.LITELLM_PRICING_URL)
            response.raise_for_status()

            data = response.json()
            pricing: dict[str, ModelPricing] = {}

            for model_name, model_data in data.items():
                if isinstance(model_data, dict):
                    try:
                        pricing[model_name] = ModelPricing(**model_data)
                    except Exception:
                        continue

            self._cached_pricing = pricing
            return pricing

    def get_model_pricing(
        self, model_name: str, pricing_data: dict[str, ModelPricing]
    ) -> ModelPricing | None:
        """
        Get pricing for a specific model, trying various name variations.

        Args:
            model_name: The model name to look up
            pricing_data: The pricing data dictionary

        Returns:
            ModelPricing if found, None otherwise
        """
        if model_name in pricing_data:
            return pricing_data[model_name]

        variations = [
            model_name,
            f"anthropic/{model_name}",
            f"claude-4-{model_name}",
            f"claude-3-5-{model_name}",
            f"claude-3-{model_name}",
            f"claude-{model_name}",
        ]

        for variant in variations:
            if variant in pricing_data:
                return pricing_data[variant]

        lower_model = model_name.lower()
        for key, value in pricing_data.items():
            if lower_model in key.lower() or key.lower() in lower_model:
                return value

        return None

    def calculate_cost_from_tokens(self, usage: TokenUsage, pricing: ModelPricing) -> float:
        """
        Calculate cost based on token usage and model pricing.

        Args:
            usage: Token usage data
            pricing: Model pricing data

        Returns:
            Total cost in USD
        """
        cost = 0.0

        if pricing.input_cost_per_token:
            cost += usage.input_tokens * pricing.input_cost_per_token

        if pricing.output_cost_per_token:
            cost += usage.output_tokens * pricing.output_cost_per_token

        if usage.cache_creation_input_tokens and pricing.cache_creation_input_token_cost:
            cost += usage.cache_creation_input_tokens * pricing.cache_creation_input_token_cost

        if usage.cache_read_input_tokens and pricing.cache_read_input_token_cost:
            cost += usage.cache_read_input_tokens * pricing.cache_read_input_token_cost

        return cost


def calculate_cost_for_entry(
    cost_usd: float | None,
    model: str | None,
    usage: TokenUsage | None,
    mode: CostMode,
    pricing_fetcher: PricingFetcher,
    pricing_data: dict[str, ModelPricing],
) -> float:
    """
    Calculate cost for a conversation entry based on the mode.

    Args:
        cost_usd: Pre-calculated cost from Claude
        model: Model name
        usage: Token usage data
        mode: Cost calculation mode
        pricing_fetcher: Pricing fetcher instance
        pricing_data: Cached pricing data

    Returns:
        Cost in USD
    """
    if mode == CostMode.DISPLAY:
        return cost_usd or 0.0

    if mode == CostMode.CALCULATE:
        if model and usage:
            pricing = pricing_fetcher.get_model_pricing(model, pricing_data)
            if pricing:
                return pricing_fetcher.calculate_cost_from_tokens(usage, pricing)
        return 0.0

    if cost_usd is not None:
        return cost_usd

    if model and usage:
        pricing = pricing_fetcher.get_model_pricing(model, pricing_data)
        if pricing:
            return pricing_fetcher.calculate_cost_from_tokens(usage, pricing)

    return 0.0

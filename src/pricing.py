from decimal import Decimal, ROUND_HALF_UP

from .schemas import Provider


# Standard uncached text prices in USD per 1 million tokens.
PRICING: dict[tuple[Provider, str], tuple[Decimal, Decimal]] = {
    ("openai", "gpt-4o-mini"): (Decimal("0.15"), Decimal("0.60")),
    ("anthropic", "claude-haiku-4-5-20251001"): (Decimal("1.00"), Decimal("5.00")),
}


def calculate_cost(
    provider: Provider, model: str, prompt_tokens: int, completion_tokens: int
) -> Decimal:
    input_rate, output_rate = PRICING[(provider, model)]
    cost = (
        prompt_tokens * input_rate + completion_tokens * output_rate
    ) / Decimal("1000000")
    return cost.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

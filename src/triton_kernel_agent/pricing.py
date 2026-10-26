"""Meta Model API token prices and per-call cost.

Source: https://dev.meta.ai/docs/pricing-rate-limits (checked 2026-10-26).
Standard mode only; Contributor mode lets Meta train on prompts, so we don't use it.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    """USD per million tokens."""

    input: float
    cached_input: float
    output: float


PRICES: dict[str, Price] = {
    "muse-wonder-1.0": Price(input=4.00, cached_input=0.20, output=20.00),
    "muse-spark-1.3": Price(input=1.25, cached_input=0.15, output=4.25),
}


def call_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float:
    """Cost of one call. cached_input_tokens is the part of input_tokens served from cache."""
    if model not in PRICES:
        raise KeyError(f"no price for model {model!r}; add it to PRICES")
    if cached_input_tokens > input_tokens:
        raise ValueError(f"cached_input_tokens {cached_input_tokens} > input_tokens {input_tokens}")
    p = PRICES[model]
    uncached = input_tokens - cached_input_tokens
    micro_usd = uncached * p.input + cached_input_tokens * p.cached_input + output_tokens * p.output
    return micro_usd / 1e6

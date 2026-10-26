import pytest

from triton_kernel_agent.pricing import call_cost_usd


def test_wonder_cost_by_hand():
    # 10k input ($4/M) + 2k output ($20/M) = 0.04 + 0.04
    assert call_cost_usd("muse-wonder-1.0", 10_000, 2_000) == pytest.approx(0.08)


def test_spark_cost_by_hand():
    # 1M input ($1.25/M) + 1M output ($4.25/M)
    assert call_cost_usd("muse-spark-1.3", 1_000_000, 1_000_000) == pytest.approx(5.50)


def test_cached_tokens_are_a_discounted_subset_of_input():
    # 10k input of which 8k cached: 2k * $4/M + 8k * $0.20/M = 0.008 + 0.0016
    cost = call_cost_usd("muse-wonder-1.0", 10_000, 0, cached_input_tokens=8_000)
    assert cost == pytest.approx(0.0096)


def test_unknown_model_raises():
    with pytest.raises(KeyError, match="no-such-model"):
        call_cost_usd("no-such-model", 1, 1)


def test_cached_more_than_input_raises():
    with pytest.raises(ValueError):
        call_cost_usd("muse-spark-1.3", 10, 0, cached_input_tokens=11)

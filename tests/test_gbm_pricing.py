import numpy as np

from asianoption import (
    arithmetic_asian_price_mc,
    arithmetic_asian_cv,
    turnbull_wakeman_arithmetic_asian_price,
)


def test_plain_mc_returns_positive_call_price():
    price, se = arithmetic_asian_price_mc(
        S0=100,
        K=100,
        r=0.05,
        T=1.0,
        sigma=0.2,
        n=12,
        n_paths=5_000,
        seed=42,
        option_type="call",
    )

    assert price > 0
    assert se > 0


def test_control_variate_has_lower_std_error_than_plain_mc():
    res = arithmetic_asian_cv(
        S0=100,
        K=100,
        r=0.05,
        T=1.0,
        sigma=0.2,
        n=12,
        n_paths=10_000,
        seed=42,
        option_type="call",
    )

    assert res.price > 0
    assert res.std_error > 0
    assert res.plain_mc_std > 0
    assert res.std_error < res.plain_mc_std
    assert res.variance_reduction > 1


def test_turnbull_wakeman_returns_positive_price():
    price = turnbull_wakeman_arithmetic_asian_price(
        S0=100,
        K=100,
        r=0.05,
        sigma=0.2,
        T=1.0,
        n=12,
        option_type="call",
    )

    assert np.isfinite(price)
    assert price > 0
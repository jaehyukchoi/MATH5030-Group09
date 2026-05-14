import numpy as np

from asianoption import (
    arithmetic_asian_heston_mc,
    arithmetic_asian_sabr_mc,
    heston_effective_vol,
    sabr_effective_vol,
)


def test_heston_effective_vol():
    sigma_eff = heston_effective_vol(
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        T=1.0,
    )

    assert np.isclose(sigma_eff, 0.2)


def test_sabr_effective_vol_beta_one():
    sigma_eff = sabr_effective_vol(
        S0=100,
        alpha0=0.2,
        beta=1.0,
    )

    assert np.isclose(sigma_eff, 0.2)


def test_heston_mc_runs():
    res = arithmetic_asian_heston_mc(
        S0=100,
        K=100,
        r=0.05,
        T=1.0,
        v0=0.04,
        kappa=2.0,
        theta=0.04,
        xi=0.3,
        rho=-0.7,
        n=12,
        n_paths=5_000,
        seed=42,
        option_type="call",
    )

    assert res.price > 0
    assert res.std_error > 0


def test_sabr_mc_runs():
    res = arithmetic_asian_sabr_mc(
        F0=100,
        K=100,
        r=0.05,
        T=1.0,
        alpha0=0.2,
        beta=1.0,
        nu=0.3,
        rho=-0.4,
        n=12,
        n_paths=5_000,
        seed=42,
        option_type="call",
    )

    assert res.price > 0
    assert res.std_error > 0
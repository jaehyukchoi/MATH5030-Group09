"""
Turnbull-Wakeman Baselines under Stochastic Volatility
=====================================================

This module provides effective-volatility Turnbull-Wakeman baselines
for arithmetic Asian options under stochastic volatility models.

The original Turnbull-Wakeman approximation is derived under GBM with
constant volatility. For stochastic volatility models, this module maps
the stochastic volatility dynamics into an effective volatility and then
uses the standard Turnbull-Wakeman formula as a fast baseline.

These functions should be interpreted as model-specific baseline
approximations, not exact stochastic-volatility Turnbull-Wakeman formulas.
"""

import numpy as np

from .approximation import turnbull_wakeman_arithmetic_asian_price


def heston_effective_vol(v0, kappa, theta, T):
    """
    Compute the effective volatility for the Heston model using the
    expected average variance over [0, T].

    The Heston variance process is

        dv_t = kappa(theta - v_t)dt + xi sqrt(v_t)dW_t.

    Since

        E[v_t] = theta + (v0 - theta) exp(-kappa t),

    the expected average variance is

        sigma_eff^2
        =
        theta + (v0 - theta) * (1 - exp(-kappa T)) / (kappa T).
    """
    if T <= 0:
        raise ValueError("T must be positive.")

    if v0 < 0 or theta < 0 or kappa < 0:
        raise ValueError("v0, theta, and kappa must be non-negative.")

    if kappa < 1e-12:
        avg_var = v0
    else:
        avg_var = theta + (v0 - theta) * (1.0 - np.exp(-kappa * T)) / (kappa * T)

    return np.sqrt(max(avg_var, 0.0))


def sabr_effective_vol(S0, alpha0, beta):
    """
    Compute a first-order effective log-volatility for the spot-style SABR model.

    The spot-style SABR model is

        dS_t = r S_t dt + alpha_t S_t^beta dW_t.

    Dividing by S_t gives

        dS_t / S_t = r dt + alpha_t S_t^(beta - 1) dW_t.

    Therefore, a simple initial local log-volatility approximation is

        sigma_eff = alpha0 * S0^(beta - 1).
    """
    if S0 <= 0:
        raise ValueError("S0 must be positive.")

    if alpha0 < 0:
        raise ValueError("alpha0 must be non-negative.")

    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta should be between 0 and 1.")

    return alpha0 * (S0 ** (beta - 1.0))


def turnbull_wakeman_heston_effective_vol_price(
    S0,
    K,
    r,
    T,
    v0,
    kappa,
    theta,
    xi,
    rho,
    n,
    option_type="call",
):
    """
    Price an arithmetic Asian option using an effective-volatility
    Turnbull-Wakeman baseline under Heston dynamics.

    The effective volatility is computed from the expected average variance
    of the Heston variance process.

    Note
    ----
    The parameters xi and rho are included for model-interface consistency,
    but this first-order effective volatility baseline does not use them
    directly. Their effects are expected to appear in the residual correction.
    """
    sigma_eff = heston_effective_vol(
        v0=v0,
        kappa=kappa,
        theta=theta,
        T=T,
    )

    return turnbull_wakeman_arithmetic_asian_price(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma_eff,
        T=T,
        n=n,
        option_type=option_type,
    )


def turnbull_wakeman_sabr_effective_vol_price(
    S0,
    K,
    r,
    T,
    alpha0,
    beta,
    nu,
    rho,
    n,
    option_type="call",
):
    """
    Price an arithmetic Asian option using an effective-volatility
    Turnbull-Wakeman baseline under SABR dynamics.

    The effective volatility is approximated by the initial local log-volatility

        sigma_eff = alpha0 * S0^(beta - 1).

    Note
    ----
    The parameters nu and rho are included for model-interface consistency,
    but this first-order effective volatility baseline does not use them
    directly. Their effects are expected to appear in the residual correction.
    """
    sigma_eff = sabr_effective_vol(
        S0=S0,
        alpha0=alpha0,
        beta=beta,
    )

    return turnbull_wakeman_arithmetic_asian_price(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma_eff,
        T=T,
        n=n,
        option_type=option_type,
    )


if __name__ == "__main__":
    from .stochastic_volatility import (
        arithmetic_asian_heston_mc,
        arithmetic_asian_sabr_mc,
    )

    # ---------------- Heston: TW effective-vol vs MC ----------------
    heston_tw = turnbull_wakeman_heston_effective_vol_price(
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
        option_type="call",
    )

    heston_mc = arithmetic_asian_heston_mc(
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
        n_paths=100_000,
        seed=42,
        option_type="call",
    )

    heston_error = heston_tw - heston_mc.price

    print("=" * 65)
    print("Heston: Effective-Vol TW vs Monte Carlo")
    print("=" * 65)
    print("Heston effective-vol TW price:", heston_tw)
    print("Heston MC price:", heston_mc.price)
    print("Heston MC std error:", heston_mc.std_error)
    print("TW - MC error:", heston_error)

    # ---------------- SABR: TW effective-vol vs MC ----------------
    sabr_tw = turnbull_wakeman_sabr_effective_vol_price(
        S0=100,
        K=100,
        r=0.05,
        T=1.0,
        alpha0=0.2,
        beta=1.0,
        nu=0.3,
        rho=-0.4,
        n=12,
        option_type="call",
    )

    sabr_mc = arithmetic_asian_sabr_mc(
        F0=100,
        K=100,
        r=0.05,
        T=1.0,
        alpha0=0.2,
        beta=1.0,
        nu=0.3,
        rho=-0.4,
        n=12,
        n_paths=100_000,
        seed=42,
        option_type="call",
    )

    sabr_error = sabr_tw - sabr_mc.price

    print("\n" + "=" * 65)
    print("SABR: Effective-Vol TW vs Monte Carlo")
    print("=" * 65)
    print("SABR effective-vol TW price:", sabr_tw)
    print("SABR MC price:", sabr_mc.price)
    print("SABR MC std error:", sabr_mc.std_error)
    print("TW - MC error:", sabr_error)
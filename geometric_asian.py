import numpy as np
from dataclasses import dataclass
import math


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / np.sqrt(2.0)))


@dataclass
class GeometricAsianResult:
    price: float
    d1: float
    d2: float
    adjusted_spot: float
    sigma_g: float
    mu_g: float


def geometric_asian_price_analytical(
    S0,
    K,
    r,
    sigma,
    T,
    n,
    option_type="call",
):
    # input check
    if S0 <= 0:
        raise ValueError("S0 must be positive.")
    if K <= 0:
        raise ValueError("K must be positive.")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    # Geometric average over n fixing dates (excluding S0):
    # G = (prod_{i=1}^n S_{t_i})^(1/n),  t_i = iT/n

    # mean and variance of ln G
    mu_lnG = np.log(S0) + (r - 0.5 * sigma**2) * T * (n + 1.0) / (2.0 * n)
    var_lnG = sigma**2 * T * (n + 1.0) * (2.0 * n + 1.0) / (6.0 * n**2)

    sigma_g = np.sqrt(var_lnG / T)
    mu_g = mu_lnG / T
    adjusted_spot = np.exp(mu_lnG + 0.5 * var_lnG)

    vol_term = np.sqrt(var_lnG)

    # zero-vol edge case
    if vol_term < 1e-14:
        G_deterministic = np.exp(mu_lnG)
        call_price = np.exp(-r * T) * max(G_deterministic - K, 0.0)
        put_price = np.exp(-r * T) * max(K - G_deterministic, 0.0)
        price = call_price if option_type == "call" else put_price

        return GeometricAsianResult(
            price=float(price),
            d1=float("nan"),
            d2=float("nan"),
            adjusted_spot=float(adjusted_spot),
            sigma_g=float(sigma_g),
            mu_g=float(mu_g),
        )

    # Black-Scholes-style formula for lognormal G
    d1 = (mu_lnG - np.log(K) + var_lnG) / vol_term
    d2 = d1 - vol_term

    discount = np.exp(-r * T)

    call_price = discount * (
        np.exp(mu_lnG + 0.5 * var_lnG) * norm_cdf(d1) - K * norm_cdf(d2)
    )

    put_price = discount * (
        K * norm_cdf(-d2) - np.exp(mu_lnG + 0.5 * var_lnG) * norm_cdf(-d1)
    )

    price = call_price if option_type == "call" else put_price

    return GeometricAsianResult(
        price=float(price),
        d1=float(d1),
        d2=float(d2),
        adjusted_spot=float(adjusted_spot),
        sigma_g=float(sigma_g),
        mu_g=float(mu_g),
    )


def geometric_asian_mc(
    S0,
    K,
    r,
    sigma,
    T,
    n,
    n_paths=100000,
    seed=42,
    option_type="call",
):
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    rng = np.random.default_rng(seed)
    dt = T / n
    Z = rng.normal(size=(n_paths, n))

    log_return = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_S = np.cumsum(log_return, axis=1)
    S = S0 * np.exp(log_S)

    # Geometric average over fixing dates only (excludes S0)
    G = np.exp(np.mean(np.log(S), axis=1))

    if option_type == "call":
        payoff = np.maximum(G - K, 0.0)
    else:
        payoff = np.maximum(K - G, 0.0)

    discounted_payoff = np.exp(-r * T) * payoff
    price = np.mean(discounted_payoff)
    std_error = np.std(discounted_payoff, ddof=1) / np.sqrt(n_paths)

    return price, std_error


if __name__ == "__main__":
    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    n = 12

    res = geometric_asian_price_analytical(S0, K, r, sigma, T, n, option_type="call")
    mc_price, mc_se = geometric_asian_mc(S0, K, r, sigma, T, n, option_type="call")

    print("Analytical price :", res.price)
    print("MC price         :", mc_price)
    print("MC std error     :", mc_se)

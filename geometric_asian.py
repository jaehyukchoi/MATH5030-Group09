import numpy as np
from dataclasses import dataclass
import math

def norm_cdf(x):
    """Standard normal CDF using numpy."""
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
    # ---- input check ----
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

    # ---- mean / variance of log G ----
    mu_lnG = np.log(S0) + (r - 0.5 * sigma**2) * T * (n + 1) / (2.0 * n)
    var_lnG = sigma**2 * T * (n + 1) * (2.0 * n + 1.0) / (6.0 * n**2)

    sigma_g = np.sqrt(var_lnG / T)
    mu_g = mu_lnG / T

    adjusted_spot = np.exp(mu_lnG + 0.5 * var_lnG)

    vol_term = np.sqrt(var_lnG)

    # ---- zero vol edge case ----
    if vol_term < 1e-14:
        forward_g = np.exp(mu_lnG)
        price_call = np.exp(-r * T) * np.maximum(forward_g - K, 0.0)
        price_put = np.exp(-r * T) * np.maximum(K - forward_g, 0.0)
        price = price_call if option_type == "call" else price_put

        return GeometricAsianResult(
            price=float(price),
            d1=np.nan,
            d2=np.nan,
            adjusted_spot=float(adjusted_spot),
            sigma_g=float(sigma_g),
            mu_g=float(mu_g),
        )

    # ---- d1, d2 ----
    d1 = (mu_lnG - np.log(K) + var_lnG) / vol_term
    d2 = d1 - vol_term

    discount = np.exp(-r * T)

    call_price = discount * (
        np.exp(mu_lnG + 0.5 * var_lnG) * norm_cdf(d1)
        - K * norm_cdf(d2)
    )

    put_price = discount * (
        K * norm_cdf(-d2)
        - np.exp(mu_lnG + 0.5 * var_lnG) * norm_cdf(-d1)
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

def geometric_asian_mc(S0, K, r, sigma, T, n, n_paths=100000, seed=42):
    rng = np.random.default_rng(seed)
    dt = T/n
    Z = rng.normal(size=(n_paths,n))

    log_return = (r-1/2*sigma**2)*dt+sigma*Z*np.sqrt(dt)
    log_S = np.cumsum(log_return,axis=1)
    S = S0*np.exp(log_S)

    #Geometric Average
    G = np.exp(np.mean(np.log(S),axis=1))
    payoff = np.maximum(G-K,0)
    price = np.exp(-r*T)*np.mean(payoff)
    return price


if __name__ == "__main__":
    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    n = 12

    res = geometric_asian_price_analytical(S0, K, r, sigma, T, n)
    mc_price = geometric_asian_mc(S0, K, r, sigma, T, n)

    print("Price:", res.price, mc_price)
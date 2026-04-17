import numpy as np
from utils import (
    norm_cdf,
    GeometricAsianResult,
    simulate_gbm_paths,
    geometric_average_mc,
    discounted,
)


def geometric_asian_price_analytical(
    S0,
    K,
    r,
    sigma,
    T,
    n,
    option_type="call",
):
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

def geometric_asian_price_mc(S0,K,r,sigma,T,n,n_paths=100000,seed = 42,option_type="call"):
    if option_type not in {"call","put"}:
        raise ValueError("optional_type must be 'call' or 'put'.")

    S_path  = simulate_gbm_paths(S0=S0,r=r,sigma=sigma,n=n,n_paths=n_paths,seed=seed,T=T)
    G = geometric_average_mc(S_path)

    if option_type == "call":
        payoff = np.maximum(G-K,0.0)
    else:
        payoff = np.maximum(K-G,0.0)

    discounted_payoff = discounted(payoff,r,T)
    price = np.mean(discounted_payoff)
    std_error = np.std(discounted_payoff,ddof = 1)/np.sqrt(n_paths)

    return price, std_error

if __name__ == "__main__":
    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    n = 12

    res = geometric_asian_price_analytical(
        S0, K, r, sigma, T, n, option_type="call"
    )
    mc_price, mc_std = geometric_asian_price_mc(
        S0, K, r, sigma, T, n, option_type="call"
    )

    print("Analytical price:", res.price)
    print("MC price        :", mc_price)
    print("MC std error    :", mc_std)
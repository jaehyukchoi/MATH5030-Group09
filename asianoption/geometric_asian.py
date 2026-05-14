import numpy as np
from .utils import (
    norm_cdf,
    GeometricAsianResult,
    discounted,
    make_fixing_times,
)


def geometric_asian_price_analytical(
    S0,
    K,
    r,
    sigma,
    T,
    n,
    option_type="call",
    averaging_start=0.0,
    averaging_end=None,
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

    # Geometric average over n fixing dates:
    #
    # Standard case:
    #   t_i = iT/n
    #
    # Flexible averaging window:
    #   t_i = T1 + i(T2 - T1)/n
    #
    # G = (prod_{i=1}^n S_{t_i})^(1/n)

    sigma_sq = sigma * sigma  # single mul, avoids pow()

    fixing_times = make_fixing_times(
        T=T,
        n=n,
        averaging_start=averaging_start,
        averaging_end=averaging_end,
    )

    mean_t = np.mean(fixing_times)
    min_matrix = np.minimum.outer(fixing_times, fixing_times)

    # mean and variance of ln G
    mu_lnG = np.log(S0) + (r - 0.5 * sigma_sq) * mean_t
    var_lnG = sigma_sq * np.sum(min_matrix) / (n * n)

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

    log_K = np.log(K)
    d1 = (mu_lnG - log_K + var_lnG) / vol_term
    d2 = d1 - vol_term

    discount = np.exp(-r * T)

    call_price = discount * (adjusted_spot * norm_cdf(d1) - K * norm_cdf(d2))
    put_price = discount * (K * norm_cdf(-d2) - adjusted_spot * norm_cdf(-d1))

    price = call_price if option_type == "call" else put_price

    return GeometricAsianResult(
        price=float(price),
        d1=float(d1),
        d2=float(d2),
        adjusted_spot=float(adjusted_spot),
        sigma_g=float(sigma_g),
        mu_g=float(mu_g),
    )


def geometric_asian_price_mc(
    S0,
    K,
    r,
    sigma,
    T,
    n,
    n_paths=100000,
    seed=42,
    option_type="call",
    Z=None,
    averaging_start=0.0,
    averaging_end=None,
):
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    fixing_times = make_fixing_times(
        T=T,
        n=n,
        averaging_start=averaging_start,
        averaging_end=averaging_end,
    )

    all_times = np.concatenate([[0.0], fixing_times])
    dt = np.diff(all_times)

    if Z is None:
        rng = np.random.default_rng(seed)
        Z = rng.normal(size=(n_paths, n))
    else:
        n_paths = Z.shape[0]

    if Z.shape[1] != n:
        raise ValueError("Z must have shape (n_paths, n).")

    drift_vec = (r - 0.5 * sigma * sigma) * dt
    vol_vec = sigma * np.sqrt(dt)

    log_inc = drift_vec[None, :] + vol_vec[None, :] * Z
    cum_log_ret = np.cumsum(log_inc, axis=1)

    log_G = np.log(S0) + np.mean(cum_log_ret, axis=1)
    G = np.exp(log_G)

    if option_type == "call":
        payoff = np.maximum(G - K, 0.0)
    else:
        payoff = np.maximum(K - G, 0.0)

    discounted_payoff = discounted(payoff, r, T)
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
    mc_price, mc_std = geometric_asian_price_mc(
        S0, K, r, sigma, T, n, option_type="call"
    )

    delayed_res = geometric_asian_price_analytical(
        S0,
        K,
        r,
        sigma,
        T,
        n,
        option_type="call",
        averaging_start=0.5,
        averaging_end=1.0,
    )

    print("Analytical price:", res.price)
    print("MC price        :", mc_price)
    print("MC std error    :", mc_std)
    print("Delayed analytical price:", delayed_res.price)
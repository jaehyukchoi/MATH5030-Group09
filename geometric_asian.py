import math
from dataclasses import dataclass


def norm_cdf(x: float) -> float:
    """Standard normal CDF using erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class GeometricAsianResult:
    price: float
    d1: float
    d2: float
    adjusted_spot: float
    sigma_g: float
    mu_g: float


def geometric_asian_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n: int,
    option_type: str = "call",
) -> GeometricAsianResult:
    """
    Price a discretely sampled geometric Asian option under GBM.

    Assumptions
    -----------
    - Under risk-neutral measure:
        dS_t = r S_t dt + sigma S_t dW_t
    - Sampling times are equally spaced:
        t_i = i * T / n,  i = 1, ..., n
    - Payoff:
        call: max(G - K, 0)
        put : max(K - G, 0)
      where G = (prod_{i=1}^n S_{t_i})^(1/n)

    Parameters
    ----------
    S0 : float
        Initial stock price
    K : float
        Strike
    r : float
        Risk-free rate
    sigma : float
        Volatility
    T : float
        Maturity
    n : int
        Number of equally spaced fixing dates
    option_type : str
        "call" or "put"

    Returns
    -------
    GeometricAsianResult
        Contains price and intermediate quantities.
    """
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

    # Mean and variance of log geometric average under risk-neutral GBM
    #
    # Let G = exp( (1/n) sum_{i=1}^n log S_{t_i} )
    # Then log G is normal with:
    #
    # mu_lnG = log(S0)
    #          + (r - 0.5*sigma^2) * T * (n + 1) / (2n)
    #
    # var_lnG = sigma^2 * T * (n + 1)(2n + 1) / (6n^2)
    #
    mu_lnG = math.log(S0) + (r - 0.5 * sigma * sigma) * T * (n + 1) / (2.0 * n)
    var_lnG = sigma * sigma * T * (n + 1) * (2.0 * n + 1.0) / (6.0 * n * n)

    sigma_g = math.sqrt(var_lnG / T) if T > 0 else 0.0
    mu_g = mu_lnG / T

    # E[G] = exp(mu_lnG + 0.5 * var_lnG)
    adjusted_spot = math.exp(mu_lnG + 0.5 * var_lnG)

    vol_term = math.sqrt(var_lnG)

    if vol_term < 1e-14:
        # Degenerate zero-vol case
        forward_g = math.exp(mu_lnG)
        discounted_call = math.exp(-r * T) * max(forward_g - K, 0.0)
        discounted_put = math.exp(-r * T) * max(K - forward_g, 0.0)
        price = discounted_call if option_type == "call" else discounted_put
        return GeometricAsianResult(
            price=price,
            d1=float("nan"),
            d2=float("nan"),
            adjusted_spot=adjusted_spot,
            sigma_g=sigma_g,
            mu_g=mu_g,
        )

    d1 = (mu_lnG - math.log(K) + var_lnG) / vol_term
    d2 = d1 - vol_term

    discounted_factor = math.exp(-r * T)

    call_price = discounted_factor * (
        math.exp(mu_lnG + 0.5 * var_lnG) * norm_cdf(d1) - K * norm_cdf(d2)
    )
    put_price = discounted_factor * (
        K * norm_cdf(-d2) - math.exp(mu_lnG + 0.5 * var_lnG) * norm_cdf(-d1)
    )

    price = call_price if option_type == "call" else put_price

    return GeometricAsianResult(
        price=price,
        d1=d1,
        d2=d2,
        adjusted_spot=adjusted_spot,
        sigma_g=sigma_g,
        mu_g=mu_g,
    )


if __name__ == "__main__":
    # Example usage
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n = 12

    call_res = geometric_asian_price(S0, K, r, sigma, T, n, option_type="call")
    put_res = geometric_asian_price(S0, K, r, sigma, T, n, option_type="put")

    print("Geometric Asian Call Price:", round(call_res.price, 6))
    print("Geometric Asian Put Price: ", round(put_res.price, 6))
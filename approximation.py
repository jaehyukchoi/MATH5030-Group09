import numpy as np
from utils import norm_cdf


def turnbull_wakeman_arithmetic_asian_price(S0, K, r, sigma, T, n, option_type="call"):
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

    dt = T / n
    t = np.arange(1, n + 1) * dt  # fixing dates t_i = i*T/n, i = 1,...,n

    # First moment: E[A] = (1/n) * sum_i S0 * exp(r * t_i)
    M1 = (S0 / n) * np.sum(np.exp(r * t))

    # Second moment: E[A^2] = (1/n^2) * sum_i sum_j E[S_{t_i} * S_{t_j}]
    # Under GBM: E[S_{t_i} * S_{t_j}] = S0^2 * exp(r*(t_i+t_j) + sigma^2*min(t_i,t_j))
    ti = t[:, None]
    tj = t[None, :]
    M2 = np.sum(S0**2 * np.exp(r * (ti + tj) + sigma**2 * np.minimum(ti, tj))) / n**2

    # Match to lognormal: ln(A) ~ N(m, v^2)
    v2 = np.log(M2) - 2.0 * np.log(M1)

    if v2 <= 0:
        intrinsic = max(M1 - K, 0.0) if option_type == "call" else max(K - M1, 0.0)
        return float(np.exp(-r * T) * intrinsic)

    v = np.sqrt(v2)
    d1 = (np.log(M1 / K) + 0.5 * v2) / v
    d2 = d1 - v

    discount = np.exp(-r * T)

    if option_type == "call":
        price = discount * (M1 * norm_cdf(d1) - K * norm_cdf(d2))
    else:
        price = discount * (K * norm_cdf(-d2) - M1 * norm_cdf(-d1))

    return float(price)

def levy_arithmetic_asian_price(

    S0,

    K,

    r,

    sigma,

    T,

    option_type="call",

):

    """

    Levy-style lognormal approximation for a continuously monitored

    arithmetic Asian option.

    Continuous average:

        A_T = (1/T) * integral_0^T S_t dt

    """

    if S0 <= 0:

        raise ValueError("S0 must be positive.")

    if K <= 0:

        raise ValueError("K must be positive.")

    if sigma < 0:

        raise ValueError("sigma must be non-negative.")

    if T <= 0:

        raise ValueError("T must be positive.")

    if option_type not in {"call", "put"}:

        raise ValueError("option_type must be 'call' or 'put'.")

    # ---- first moment m1 = E[A_T] ----

    if abs(r) < 1e-12:

        m1 = S0

    else:

        m1 = S0 * (np.exp(r * T) - 1.0) / (r * T)

    # ---- second moment m2 = E[A_T^2] ----

    # m2 = (2 S0^2 / (T^2 (r+sigma^2))) *

    #      [ (exp((2r+sigma^2)T)-1)/(2r+sigma^2) - (exp(rT)-1)/r ]

    #

    # handle r ~ 0 carefully

    a = r + sigma**2

    b = 2.0 * r + sigma**2

    if abs(r) < 1e-12:

        m2 = (2.0 * S0**2 / (T**2 * a)) * (

            (np.exp(b * T) - 1.0) / b - T

        )

    else:

        m2 = (2.0 * S0**2 / (T**2 * a)) * (

            (np.exp(b * T) - 1.0) / b

            - (np.exp(r * T) - 1.0) / r

        )

    # ---- lognormal matching ----

    ratio = m2 / (m1**2)

    ratio = max(ratio, 1.0)

    sigma_lnA_sq = np.log(ratio)

    if sigma_lnA_sq <= 1e-14:

        intrinsic = max(m1 - K, 0.0) if option_type == "call" else max(K - m1, 0.0)

        return float(np.exp(-r * T) * intrinsic)

    sigma_lnA = np.sqrt(sigma_lnA_sq)

    d1 = (np.log(m1 / K) + 0.5 * sigma_lnA_sq) / sigma_lnA

    d2 = d1 - sigma_lnA

    discount = np.exp(-r * T)

    if option_type == "call":

        price = discount * (m1 * norm_cdf(d1) - K * norm_cdf(d2))

    else:

        price = discount * (K * norm_cdf(-d2) - m1 * norm_cdf(-d1))

    return float(price)

if __name__ == "__main__":

    S0 = 100

    K = 100

    r = 0.05

    sigma = 0.2

    T = 1.0

    n = 12

    tw_call = turnbull_wakeman_arithmetic_asian_price(

        S0, K, r, sigma, T, n, option_type="call"

    )

    tw_put = turnbull_wakeman_arithmetic_asian_price(

        S0, K, r, sigma, T, n, option_type="put"

    )

    levy_call = levy_arithmetic_asian_price(

        S0, K, r, sigma, T, option_type="call"

    )

    levy_put = levy_arithmetic_asian_price(

        S0, K, r, sigma, T, option_type="put"

    )

    print(f"TW call price   : {tw_call:.6f}")

    print(f"TW put price    : {tw_put:.6f}")

    print(f"Levy call price : {levy_call:.6f}")

    print(f"Levy put price  : {levy_put:.6f}")


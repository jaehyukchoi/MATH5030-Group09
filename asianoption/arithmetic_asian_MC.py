import numpy as np

from .utils import discounted
from .utils import make_fixing_times


def arithmetic_asian_price_mc(
    S0,
    K,
    r,
    T,
    sigma,
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

    S = np.full(n_paths, S0, dtype=float)
    sum_S = np.zeros(n_paths, dtype=float)

    for j in range(n):
        S = S * np.exp(
            (r - 0.5 * sigma * sigma) * dt[j]
            + sigma * np.sqrt(dt[j]) * Z[:, j]
        )
        sum_S += S

    average = sum_S / n

    if option_type == "call":
        payoff = np.maximum(average - K, 0.0)
    else:
        payoff = np.maximum(K - average, 0.0)

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

    price, std = arithmetic_asian_price_mc(
        S0,
        K,
        r,
        T,
        sigma,
        n,
    )

    print("Standard [0,T] averaging:", price, std)

    delayed_price, delayed_std = arithmetic_asian_price_mc(
        S0,
        K,
        r,
        T,
        sigma,
        n,
        averaging_start=0.5,
        averaging_end=1.0,
    )

    print("Delayed [0.5,T] averaging:", delayed_price, delayed_std)
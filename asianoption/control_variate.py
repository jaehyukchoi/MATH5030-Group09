"""
Control Variate Method for Arithmetic Asian Option Pricing
==========================================================

Uses the geometric Asian option as a control variate to reduce the
variance of the Monte Carlo estimate for the arithmetic Asian option.

The key idea:
    - The geometric Asian option has a known closed-form price (E[X]).
    - On the SAME simulated paths we compute both the arithmetic payoff (Y)
      and the geometric payoff (X).
    - Because they are highly correlated, the adjusted estimator
          theta_CV = mean(Y) - beta * (mean(X) - E[X])
      has much lower variance than the plain Monte Carlo estimator mean(Y).
    - The optimal beta is  Cov(Y, X) / Var(X).

Computational Optimization:
    - The geometric average is computed entirely in log-space, reusing the
      cumulative log-returns that are already available from the GBM
      simulation.  This avoids a redundant exp -> log round-trip on the
      full (n_paths x n) matrix.
    - Array concatenation for prepending S0 is replaced by direct scalar
      arithmetic, eliminating a temporary (n_paths x 1) allocation.
    - All scalar constants (drift, vol*sqrt(dt), discount factor, log(S0))
      are pre-computed once before the vectorised loop.

Reference:
    Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering.
    Springer. Chapter 4 -- Variance Reduction Techniques.
"""

import numpy as np
from dataclasses import dataclass
from .geometric_asian import geometric_asian_price_analytical
from .utils import make_fixing_times


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class ControlVariateResult:
    """Stores the output of the control variate pricing method."""

    price: float  # CV-adjusted arithmetic Asian price
    std_error: float  # standard error of the CV estimator
    beta: float  # optimal control variate coefficient
    rho: float
    plain_mc_price: float  # plain MC price (no CV adjustment)
    plain_mc_std: float  # plain MC standard error
    geo_analytical: float  # geometric closed-form price used as E[X]
    variance_reduction: float  # ratio: plain_var / cv_var  (> 1 is good)


# ---------------------------------------------------------------------------
# Main pricing function
# ---------------------------------------------------------------------------
def arithmetic_asian_cv(
    S0,
    K,
    r,
    T,
    sigma,
    n,
    n_paths=100_000,
    seed=42,
    option_type="call",
    Z=None,
    averaging_start=0.0,
    averaging_end=None,
):
    """
    Price an arithmetic Asian option using Monte Carlo with a geometric
    Asian control variate.

    Parameters
    ----------
    S0 : float
        Initial stock price.
    K : float
        Strike price.
    r : float
        Risk-free interest rate (annualized, continuous compounding).
    T : float
        Time to maturity in years.
    sigma : float
        Volatility of the underlying (annualized).
    n : int
        Number of monitoring dates.
    n_paths : int, optional
        Number of Monte Carlo paths (default 100,000).
    seed : int, optional
        Random seed for reproducibility (default 42).
    option_type : str, optional
        'call' or 'put' (default 'call').
    averaging_start : float, optional
        Start time of the averaging window. Default is 0.0.
    averaging_end : float or None, optional
        End time of the averaging window. If None, defaults to T.

    Returns
    -------
    ControlVariateResult
        Dataclass with the CV price, standard error, beta, and diagnostics.
    """
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    # ------------------------------------------------------------------
    # Pre-compute scalar constants (evaluated once, reused across all paths)
    # ------------------------------------------------------------------
    fixing_times = make_fixing_times(
        T=T,
        n=n,
        averaging_start=averaging_start,
        averaging_end=averaging_end,
    )

    all_times = np.concatenate([[0.0], fixing_times])
    dt = np.diff(all_times)

    drift = (r - 0.5 * sigma * sigma) * dt  # sigma*sigma: single mul, avoids pow()
    vol_sqrt_dt = sigma * np.sqrt(dt)  # one sqrt; reused for every path & step
    discount = np.exp(-r * T)  # one exp; applied to all payoffs below
    log_S0 = np.log(S0)  # one log; reused in geometric avg (Step 4)
    inv_n = 1.0 / n  # reciprocal: mul is cheaper than div

    # ------------------------------------------------------------------
    # Step 1 -- Generate log-return increments
    #
    # log_inc[i, j] = log(S_{t_{j+1}} / S_{t_j})  for path i
    # ------------------------------------------------------------------
    if Z is None:
        rng = np.random.default_rng(seed)
        Z = rng.normal(size=(n_paths, n))
    else:
        n_paths = Z.shape[0]  # infer from provided Z to keep std_error correct

    if Z.shape[1] != n:
        raise ValueError("Z must have shape (n_paths, n).")

    log_inc = drift[None, :] + vol_sqrt_dt[None, :] * Z  # shape (n_paths, n)

    # ------------------------------------------------------------------
    # Step 2 -- Cumulative log-returns (addition only, no transcendentals)
    #
    # cum_log_ret[i, j] = log(S_{t_{j+1}} / S0)
    # ------------------------------------------------------------------
    cum_log_ret = np.cumsum(log_inc, axis=1)  # O(n_paths * n) additions

    # ------------------------------------------------------------------
    # Step 3 -- Arithmetic average  (requires actual prices; one exp call)
    #
    # A = (S_{t1} + ... + S_{tn}) / n
    #   = S0 * sum_j exp(clr_j) / n
    #
    # Factoring out S0 lets exp operate on centred log-returns (near zero),
    # which is more numerically stable than exponentiating absolute
    # log-prices.
    # ------------------------------------------------------------------
    S_ratio = np.exp(cum_log_ret)  # sole large-matrix exp in the function
    arith_avg = (
        S0 * np.sum(S_ratio, axis=1) * inv_n
    )  # mul by inv_n avoids division per element

    # ------------------------------------------------------------------
    # Step 4 -- Geometric average  (stays in log-space, no extra exp/log)
    #
    # log(G) = (1/n) * sum_{i=1}^{n} log(S_{t_i})
    #        = (1/n) * [n*log(S0) + sum_{j=1}^{n} clr_j]
    #        = log(S0) + sum(cum_log_ret, axis=1) / n
    #
    # This reuses cum_log_ret directly, avoiding a redundant exp -> log
    # round-trip on the full (n_paths x n) price matrix.  The only
    # transcendental call is a single exp on the resulting 1-D vector.
    # ------------------------------------------------------------------
    log_geo_avg = (
        log_S0 + np.sum(cum_log_ret, axis=1) * inv_n
    )  # all additions; no exp/log on matrix
    geo_avg = np.exp(log_geo_avg)  # exp on 1-D vector (n_paths,) only

    # ------------------------------------------------------------------
    # Step 5 -- Discounted payoffs
    # ------------------------------------------------------------------
    if option_type == "call":
        Y = discount * np.maximum(arith_avg - K, 0.0)  # arithmetic
        X = discount * np.maximum(geo_avg - K, 0.0)  # geometric
    else:
        Y = discount * np.maximum(K - arith_avg, 0.0)
        X = discount * np.maximum(K - geo_avg, 0.0)

    # ------------------------------------------------------------------
    # Step 6 -- Known geometric price from the closed-form formula
    # ------------------------------------------------------------------
    geo_result = geometric_asian_price_analytical(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n=n,
        option_type=option_type,
        averaging_start=averaging_start,
        averaging_end=averaging_end,
    )
    E_X = geo_result.price

    # ------------------------------------------------------------------
    # Step 7 -- Correlation check and optimal beta
    #
    # In finance, before using a control variate, we first check whether
    # the candidate control variable X is highly correlated with the target
    # variable Y.
    #
    # Here:
    #   Y = discounted arithmetic Asian payoff
    #   X = discounted geometric Asian payoff
    #
    # The sample correlation is:
    #
    #   rho = Corr(Y, X)
    #
    # If |rho| is close to 1, X is a good control variate for Y.
    #
    # The optimal control variate coefficient is:
    #
    #   beta* = Cov(Y, X) / Var(X)
    #
    # Equivalently:
    #
    #   beta* = rho * std(Y) / std(X)
    #
    # In this implementation, rho is estimated from the same simulated paths.
    # ------------------------------------------------------------------
    mean_Y = np.mean(Y)
    mean_X = np.mean(X)

    Y_c = Y - mean_Y
    X_c = X - mean_X

    cov_YX = Y_c @ X_c
    var_Y = Y_c @ Y_c
    var_X = X_c @ X_c

    if var_Y < 1e-20 or var_X < 1e-20:
        rho = 0.0
    else:
        rho = cov_YX / np.sqrt(var_Y * var_X)

    if var_X < 1e-20:
        beta = 0.0
    else:
        beta = cov_YX / var_X

    # ------------------------------------------------------------------
    # Step 8 -- Control-variate adjusted estimator
    # ------------------------------------------------------------------
    Y_cv = Y - beta * (X - E_X)

    cv_price = np.mean(Y_cv)
    cv_std = np.std(Y_cv, ddof=1) / np.sqrt(n_paths)

    # ------------------------------------------------------------------
    # Plain MC stats (for comparison / variance-reduction ratio)
    # ------------------------------------------------------------------
    plain_price = mean_Y
    plain_std = np.std(Y, ddof=1) / np.sqrt(n_paths)

    if cv_std > 0:
        var_reduction = (plain_std / cv_std) ** 2
    else:
        var_reduction = float("inf")

    return ControlVariateResult(
        price=cv_price,
        std_error=cv_std,
        beta=beta,
        rho=rho,
        plain_mc_price=plain_price,
        plain_mc_std=plain_std,
        geo_analytical=E_X,
        variance_reduction=var_reduction,
    )


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    n = 12

    print("=" * 65)
    print("  Control Variate: Arithmetic Asian Option Pricing")
    print("=" * 65)
    print(f"  S0={S0}  K={K}  r={r}  sigma={sigma}  T={T}  n={n}")
    print("-" * 65)

    for opt in ("call", "put"):
        res = arithmetic_asian_cv(
            S0,
            K,
            r,
            T,
            sigma,
            n,
            n_paths=500_000,
            option_type=opt,
        )

        print(f"\n  Option type : {opt}")
        print(f"  CV price    : {res.price:.6f}")
        print(f"  CV std err  : {res.std_error:.6f}")
        print(f"  Rho         : {res.rho:.6f}")
        print(f"  Plain MC    : {res.plain_mc_price:.6f}  (std {res.plain_mc_std:.6f})")
        print(f"  Geo analyt. : {res.geo_analytical:.6f}")
        print(f"  Beta        : {res.beta:.4f}")
        print(f"  Var reduction: {res.variance_reduction:.1f}x")

    print("\nDelayed averaging window [0.5, 1.0]")

    res_delayed = arithmetic_asian_cv(
        S0,
        K,
        r,
        T,
        sigma,
        n,
        n_paths=500_000,
        option_type="call",
        averaging_start=0.5,
        averaging_end=1.0,
    )

    print(f"  CV delayed price : {res_delayed.price:.6f}")
    print(f"  CV delayed stderr: {res_delayed.std_error:.6f}")
    print(f"  Rho              : {res_delayed.rho:.6f}")
    print(f"  Beta             : {res_delayed.beta:.4f}")

    print("\n" + "=" * 65)
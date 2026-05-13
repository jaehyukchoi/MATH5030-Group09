"""
Stochastic Volatility Models for Asian Option Pricing
====================================================
This module implements Monte Carlo pricing for arithmetic Asian options
under stochastic volatility models.
The first implemented model is the Heston model:
    dS_t = r S_t dt + sqrt(v_t) S_t dW_t^S
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW_t^v
with
    Corr(dW_t^S, dW_t^v) = rho.
The arithmetic Asian payoff is based on the discrete average:
    A = (1/n) sum_{i=1}^n S_{t_i}.
This module is intended as a stochastic-volatility extension of the
Black-Scholes GBM Asian option pricing framework.
"""

from dataclasses import dataclass


import numpy as np

@dataclass
class HestonAsianMCResult:
    price: float
    std_error:float
    terminal_mean: float
    average_mean: float
    variance_mean: float

@dataclass

class SABRAsianMCResult:
    price: float
    std_error: float
    terminal_mean: float
    average_mean: float
    alpha_terminal_mean: float



def arithmetic_asian_sabr_mc(
        F0,
        K,
        r,
        T,
        alpha0,
        beta,
        nu,
        rho,
        n,
        n_paths=100000,
        seed=42,
        option_type="call",
        log_euler=True,
):
    """
    Price an arithmetic Asian option under the SABR stochastic volatility model.

    The SABR model is

        dF_t = alpha_t F_t^beta dW_t^F

        d alpha_t = nu alpha_t dW_t^alpha

    with

        Corr(dW_t^F, dW_t^alpha) = rho.

    The arithmetic Asian payoff is based on the discrete average:

        A = (1/n) sum_{i=1}^n F_{t_i}.

    Parameters
    ----------
    F0 : float
        Initial forward or underlying level.

    K : float
        Strike price.

    r : float
        Risk-free rate used for discounting.

    T : float
        Time to maturity.

    alpha0 : float
        Initial stochastic volatility level.

    beta : float
        SABR elasticity parameter. Usually beta is between 0 and 1.

    nu : float
        Volatility of volatility.

    rho : float
        Correlation between the price Brownian motion and volatility Brownian motion.

    n : int
        Number of monitoring dates.

    n_paths : int, optional
        Number of Monte Carlo paths.

    seed : int, optional
        Random seed.

    option_type : str, optional
        "call" or "put".

    log_euler : bool, optional
        If True, uses a log-Euler style update for the underlying.
        This helps preserve positivity.

    Returns
    -------
    SABRAsianMCResult
        Price, standard error, and basic path diagnostics.
    """
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'.")

    if not -1.0 <= rho <= 1.0:
        raise ValueError("rho must be between -1 and 1.")

    if F0 <= 0 or K <= 0 or T <= 0 or n <= 0 or n_paths <= 0:
        raise ValueError("F0, K, T, n, and n_paths must be positive.")

    if alpha0 < 0 or nu < 0:
        raise ValueError("alpha0 and nu must be non-negative.")

    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta should be between 0 and 1.")

    dt = T / n
    sqrt_dt = np.sqrt(dt)
    discount = np.exp(-r * T)

    rng = np.random.default_rng(seed)

    F = np.full(n_paths, F0, dtype=float)
    alpha = np.full(n_paths, alpha0, dtype=float)

    sum_F = np.zeros(n_paths, dtype=float)

    for _ in range(n):
        Z_alpha = rng.normal(size=n_paths)
        Z_2 = rng.normal(size=n_paths)

        Z_F = rho * Z_alpha + np.sqrt(1.0 - rho * rho) * Z_2

        # SABR volatility process:
        # d alpha_t = nu alpha_t dW_t
        # lognormal exact step under Euler time grid
        alpha = alpha * np.exp(
            -0.5 * nu * nu * dt + nu * sqrt_dt * Z_alpha
        )

        F_pos = np.maximum(F, 1e-12)

        if log_euler:
            # Approximate log-Euler update:
            #
            # dF / F = alpha F^(beta - 1) dW
            #
            # local instantaneous volatility of log(F):
            # sigma_log = alpha * F^(beta - 1)
            sigma_log = alpha * np.power(F_pos, beta - 1.0)

            F = F_pos * np.exp(
                (r - 0.5 * sigma_log * sigma_log) * dt
                + sigma_log * sqrt_dt * Z_F
            )
        else:
            # Direct Euler update:
            #
            # F_{t+dt} = F_t + alpha F_t^beta dW
            #
            # This is simpler but can produce negative F for beta < 1.
            F = F + r * F_pos * dt + alpha * np.power(F_pos, beta) * sqrt_dt * Z_F
            F = np.maximum(F, 1e-12)

        sum_F += F

    arithmetic_avg = sum_F / n

    if option_type == "call":
        payoff = np.maximum(arithmetic_avg - K, 0.0)
    else:
        payoff = np.maximum(K - arithmetic_avg, 0.0)

    discounted_payoff = discount * payoff

    price = np.mean(discounted_payoff)
    std_error = np.std(discounted_payoff, ddof=1) / np.sqrt(n_paths)

    return SABRAsianMCResult(
        price=price,
        std_error=std_error,
        terminal_mean=np.mean(F),
        average_mean=np.mean(arithmetic_avg),
        alpha_terminal_mean=np.mean(alpha),
    )

def arithmetic_asian_heston_mc(
        S0,K,r,T,v0,kappa,theta,xi,rho,n,n_paths = 100000,seed=42,option_type="call",full_truncation=True,
):
    """
    Price an arithmetic Asian option under the Heston stochastic volatility model.
    Parameters
    ----------
    S0 : float
        Initial stock price.

    K : float
        Strike price.

    r : float
        Risk-free rate.

    T : float
        Time to maturity.

    v0 : float
        Initial variance.

    kappa : float
        Mean-reversion speed of variance.

    theta : float
        Long-run variance level.

    xi : float
        Volatility of variance.

    rho : float
        Correlation between stock and variance Brownian motions.

    n : int
        Number of monitoring dates.

    n_paths : int, optional
        Number of Monte Carlo paths.

    seed : int, optional
        Random seed.

    option_type : str, optional
        "call" or "put".

    full_truncation : bool, optional
        If True, uses full truncation Euler to keep variance non-negative.

     Returns

    -------
    HestonAsianMCResult
        Price, standard error, and basic path diagnostics.
    """
    if option_type not in {"call","put"}:
        raise ValueError("option_type must be 'call' or 'put")
    if not -1 <= rho<=1:
        raise ValueError("rho must be between -1 and 1")
    if S0 <= 0 or K <= 0 or T <= 0 or n <= 0 or n_paths <= 0:
        raise ValueError("S0, K, T, n, and n_paths must be positive.")
    if v0 < 0 or theta < 0 or kappa < 0 or xi < 0:
        raise ValueError("v0, theta, kappa, and xi must be non-negative.")

    dt = T/n
    sqrt_dt = np.sqrt(dt)
    discount = np.exp(-r*T)

    rng = np.random.default_rng(seed)

    S = np.full(n_paths,S0,dtype=float)
    v = np.full(n_paths,v0,dtype=float)

    sum_S = np.zeros(n_paths,dtype = float)

    for _ in range(n):
        Z1 = rng.normal(size = n_paths)
        Z2 = rng.normal(size = n_paths)

        dW_v = sqrt_dt * Z1
        dW_s = sqrt_dt*(rho*Z1+np.sqrt(1.0-rho*rho)*Z2)

        if full_truncation:
            v_pos = np.maximum(v, 0.0)
        #ensure the variance is always non-negative
        else:
            v_pos = v

        #variance process
        v_next =(
            v+kappa*(theta-v_pos)*dt+xi*np.sqrt(v_pos)*dW_v
        )

        if full_truncation:
            v_next = np.maximum(v_next, 0.0)

        #underlying asset price process
        S = S*np.exp((r-0.5*v_pos)*dt+np.sqrt(v_pos)*dW_s)

        v = v_next
        sum_S += S

    arithmetic_avg = sum_S /n

    if option_type == "call":
        payoff = np.maximum(arithmetic_avg-K,0.0)
    else:
        payoff = np.maximum(K - arithmetic_avg, 0.0)

    discounted_payoff = discount * payoff
    price = np.mean(discounted_payoff)
    std_error = np.std(discounted_payoff, ddof=1) / np.sqrt(n_paths)

    return HestonAsianMCResult(
        price=price,
        std_error=std_error,
        terminal_mean=np.mean(S),
        average_mean=np.mean(arithmetic_avg),
        variance_mean=np.mean(v),
    )

if __name__ == "__main__":

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
        n_paths=100_000,
        seed=42,
        option_type="call",
    )

    print("Heston Asian MC price:", res.price)
    print("Std error:", res.std_error)
    print("Mean terminal S:", res.terminal_mean)
    print("Mean arithmetic average:", res.average_mean)
    print("Mean terminal variance:", res.variance_mean)
    print("\n" + "=" * 65)
    print("SABR Asian Monte Carlo Demo")
    print("=" * 65)

    sabr_res = arithmetic_asian_sabr_mc(
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

    print("SABR Asian MC price:", sabr_res.price)
    print("Std error:", sabr_res.std_error)
    print("Mean terminal F:", sabr_res.terminal_mean)
    print("Mean arithmetic average:", sabr_res.average_mean)
    print("Mean terminal alpha:", sabr_res.alpha_terminal_mean)















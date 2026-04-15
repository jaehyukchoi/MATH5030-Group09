import numpy as np

from utils import (
    simulate_gbm_paths,
    arithmetic_average_mc,
    geometric_average_mc,
    discounted,
)

from geometric_asian import geometric_asian_price_analytical

def arithmetic_asian_price_mc_cv(S0,K,r,T,sigma,n,n_paths=100000,seed=42,option_type="call"):
    """

    Monte Carlo pricing for arithmetic Asian option using geometric Asian option as a control variate

    The averaging convention includes S0:
        A = (S0 + S_t1 + ... + S_tn) / (n+1)
        G = (S0 * S_t1 * ... * S_tn)^(1/(n+1))

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

    S_path = simulate_gbm_paths(S0,r=r,sigma=sigma,T=T,n=n,n_paths=n_paths, seed=seed)
    A = arithmetic_average_mc(S_path)

    if option_type == "call":
        payoff_arith = np.maximum(A - K, 0.0)
    else:
        payoff_arith = np.maximum(K - A, 0.0)

    X = discounted(payoff_arith, r, T)

    #geometric average as control payoff
    G = geometric_average_mc(S_path)

    G = geometric_average_mc(S_path)

    if option_type == "call":
        payoff_geo = np.maximum(G - K, 0.0)
    else:
        payoff_geo = np.maximum(K - G, 0.0)

    Y = discounted(payoff_geo, r, T)

    geo_exact= geometric_asian_price_analytical(S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n=n,
        option_type=option_type,
    ).price

    #without control variate
    plain_price = np.mean(X)
    plain_std_error = np.std(X,ddof=1)/np.sqrt(n_paths)

    #optimal control coefficient b*
    var_Y = np.var(Y,ddof=1)
    if var_Y <1e-14:
        b_opt = 0.0
    else:
        cov_XY = np.cov(X,Y,ddof=1)[0,1]

        b_opt =cov_XY/var_Y

    X_cv = X-b_opt*(Y-geo_exact)
    cv_price = np.mean(X_cv)
    cv_std_error = np.std(X_cv,ddof=1)/np.sqrt(n_paths)

    return {
        "plain_price": float(plain_price),
        "plain_std_error": float(plain_std_error),
        "cv_price": float(cv_price),
        "cv_std_error": float(cv_std_error),
        "b_opt": float(b_opt),
        "geo_exact": float(geo_exact),
    }

if __name__ == "__main__":
    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    n = 12

    result = arithmetic_asian_price_mc_cv(
        S0=S0,
        K=K,
        r=r,
        T=T,
        sigma=sigma,
        n=n,
        n_paths=100000,
        seed=42,
        option_type="call",
    )

    print("Plain MC price      :", result["plain_price"])
    print("Plain MC std error  :", result["plain_std_error"])
    print("CV MC price         :", result["cv_price"])
    print("CV MC std error     :", result["cv_std_error"])
    print("Optimal b           :", result["b_opt"])
    print("Geometric exact     :", result["geo_exact"])



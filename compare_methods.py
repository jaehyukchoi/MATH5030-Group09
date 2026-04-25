import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arithmetic_asian_MC import arithmetic_asian_price_mc
from control_variate import arithmetic_asian_cv
from approximation import (
    turnbull_wakeman_arithmetic_asian_price,
    levy_arithmetic_asian_price,
)


def compare_one_case(
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
    """
    Compare pricing methods for one parameter set.

    Benchmark in this script:
        Plain MC with the same random shocks Z.

    Errors are therefore:
        method price - plain MC price

    This is a deviation-from-MC study, not a true-pricing-error study.
    """
    result = {
        "S0": S0,
        "K": K,
        "r": r,
        "sigma": sigma,
        "T": T,
        "n": n,
        "option_type": option_type,
    }

    # Common random numbers for Plain MC and CV MC
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n_paths, n))

    # ---------------- Plain MC ----------------
    t0 = time.perf_counter()
    mc_price, mc_se = arithmetic_asian_price_mc(
        S0=S0,
        K=K,
        r=r,
        T=T,
        sigma=sigma,
        n=n,
        n_paths=n_paths,
        seed=seed,
        option_type=option_type,
        Z=Z,
    )
    t1 = time.perf_counter()

    result["mc_price"] = mc_price
    result["mc_se"] = mc_se
    result["mc_time"] = t1 - t0

    # ---------------- Control Variate MC ----------------
    t0 = time.perf_counter()
    cv_res = arithmetic_asian_cv(
        S0=S0,
        K=K,
        r=r,
        T=T,
        sigma=sigma,
        n=n,
        n_paths=n_paths,
        seed=seed,
        option_type=option_type,
        Z=Z,
    )
    t1 = time.perf_counter()

    result["cv_price"] = cv_res.price
    result["cv_se"] = cv_res.std_error
    result["cv_beta"] = cv_res.beta
    result["cv_plain_mc_price"] = cv_res.plain_mc_price
    result["cv_plain_mc_std"] = cv_res.plain_mc_std
    result["geo_analytical"] = cv_res.geo_analytical
    result["variance_reduction"] = cv_res.variance_reduction
    result["cv_time"] = t1 - t0

    # Check whether standalone MC and CV-internal plain MC match
    result["mc_cv_plain_diff"] = result["mc_price"] - result["cv_plain_mc_price"]
    result["mc_cv_std_diff"] = result["mc_se"] - result["cv_plain_mc_std"]

    # ---------------- Turnbull-Wakeman ----------------
    t0 = time.perf_counter()
    tw_price = turnbull_wakeman_arithmetic_asian_price(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n=n,
        option_type=option_type,
    )
    t1 = time.perf_counter()

    result["tw_price"] = tw_price
    result["tw_time"] = t1 - t0

    # ---------------- Levy ----------------
    t0 = time.perf_counter()
    levy_price = levy_arithmetic_asian_price(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        option_type=option_type,
    )
    t1 = time.perf_counter()

    result["levy_price"] = levy_price
    result["levy_time"] = t1 - t0

    # ---------------- Deviations from Plain MC ----------------
    result["cv_error"] = result["cv_price"] - result["mc_price"]
    result["tw_error"] = result["tw_price"] - result["mc_price"]
    result["levy_error"] = result["levy_price"] - result["mc_price"]

    result["cv_abs_error"] = abs(result["cv_error"])
    result["tw_abs_error"] = abs(result["tw_error"])
    result["levy_abs_error"] = abs(result["levy_error"])

    return result


def run_grid_experiment(
    S0=100,
    r=0.05,
    T=1.0,
    K_list=None,
    sigma_list=None,
    n_list=None,
    n_paths=100000,
    seed=42,
    option_type="call",
):
    if K_list is None:
        K_list = [100]
    if sigma_list is None:
        sigma_list = [0.2]
    if n_list is None:
        n_list = [4, 8, 12, 26, 52, 126, 252]

    rows = []

    for K in K_list:
        for sigma in sigma_list:
            for n in n_list:
                row = compare_one_case(
                    S0=S0,
                    K=K,
                    r=r,
                    sigma=sigma,
                    T=T,
                    n=n,
                    n_paths=n_paths,
                    seed=seed,
                    option_type=option_type,
                )
                rows.append(row)

    return pd.DataFrame(rows)


def filter_case(df, K_fixed=100, sigma_fixed=0.2, option_type="call"):
    return df[
        (df["K"] == K_fixed)
        & (df["sigma"] == sigma_fixed)
        & (df["option_type"] == option_type)
    ].sort_values("n")


def plot_price_vs_n(df, K_fixed=100, sigma_fixed=0.2, option_type="call"):
    sub = filter_case(df, K_fixed, sigma_fixed, option_type)

    plt.figure(figsize=(8, 5))
    plt.plot(sub["n"], sub["mc_price"], marker="o", label="Plain MC")
    plt.plot(sub["n"], sub["cv_price"], marker="o", label="CV MC")
    plt.plot(sub["n"], sub["tw_price"], marker="o", label="Turnbull-Wakeman")
    plt.plot(sub["n"], sub["levy_price"], marker="o", label="Levy continuous")

    plt.xlabel("Number of monitoring dates n")
    plt.ylabel("Option price")
    plt.title(
        f"Price vs Monitoring Frequency "
        f"(K={K_fixed}, sigma={sigma_fixed}, type={option_type})"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_error_vs_n(df, K_fixed=100, sigma_fixed=0.2, option_type="call"):
    sub = filter_case(df, K_fixed, sigma_fixed, option_type)

    plt.figure(figsize=(8, 5))
    plt.plot(sub["n"], sub["cv_error"], marker="o", label="CV MC - MC")
    plt.plot(sub["n"], sub["tw_error"], marker="o", label="TW - MC")
    plt.plot(sub["n"], sub["levy_error"], marker="o", label="Levy - MC")

    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Number of monitoring dates n")
    plt.ylabel("Method price - Plain MC price")
    plt.title(
        f"Deviation vs Monitoring Frequency "
        f"(K={K_fixed}, sigma={sigma_fixed}, type={option_type})"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_runtime_vs_n(df, K_fixed=100, sigma_fixed=0.2, option_type="call"):
    sub = filter_case(df, K_fixed, sigma_fixed, option_type)

    plt.figure(figsize=(8, 5))
    plt.plot(sub["n"], sub["mc_time"], marker="o", label="Plain MC")
    plt.plot(sub["n"], sub["cv_time"], marker="o", label="CV MC")
    plt.plot(sub["n"], sub["tw_time"], marker="o", label="Turnbull-Wakeman")
    plt.plot(sub["n"], sub["levy_time"], marker="o", label="Levy")

    plt.xlabel("Number of monitoring dates n")
    plt.ylabel("Runtime seconds")
    plt.title(
        f"Runtime vs Monitoring Frequency "
        f"(K={K_fixed}, sigma={sigma_fixed}, type={option_type})"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_summary(df):
    cols = [
        "K",
        "sigma",
        "n",
        "mc_price",
        "mc_se",
        "cv_price",
        "cv_se",
        "variance_reduction",
        "tw_price",
        "levy_price",
        "cv_error",
        "tw_error",
        "levy_error",
        "mc_cv_plain_diff",
    ]

    print("\nMonitoring frequency study summary:\n")
    print(df[cols].round(6))


if __name__ == "__main__":
    n_list = [4, 6, 8, 12, 18, 26, 36, 52, 78, 126, 180, 252]

    df = run_grid_experiment(
        S0=100,
        r=0.05,
        T=1.0,
        K_list=[100],
        sigma_list=[0.2],
        n_list=n_list,
        n_paths=100000,
        seed=42,
        option_type="call",
    )

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print_summary(df)

    plot_price_vs_n(df, K_fixed=100, sigma_fixed=0.2, option_type="call")
    plot_error_vs_n(df, K_fixed=100, sigma_fixed=0.2, option_type="call")
    plot_runtime_vs_n(df, K_fixed=100, sigma_fixed=0.2, option_type="call")
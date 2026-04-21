import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
    result = {
        "S0": S0,
        "K": K,
        "r": r,
        "sigma": sigma,
        "T": T,
        "n": n,
        "option_type": option_type,
    }
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

    # ---------------- Errors vs plain MC baseline ----------------
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
        K_list = [80, 90, 100, 110, 120]
    if sigma_list is None:
        sigma_list = [0.1, 0.2, 0.4]
    if n_list is None:
        n_list = [12, 52, 252]

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


def plot_error_vs_strike(df, sigma_fixed=0.2, n_fixed=12, option_type="call"):
    sub = df[
        (df["sigma"] == sigma_fixed)
        & (df["n"] == n_fixed)
        & (df["option_type"] == option_type)
    ].sort_values("K")

    plt.figure(figsize=(8, 5))
    plt.plot(sub["K"], sub["cv_error"], marker="o", label="CV MC - MC")
    plt.plot(sub["K"], sub["tw_error"], marker="o", label="TW - MC")
    plt.plot(sub["K"], sub["levy_error"], marker="o", label="Levy - MC")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Strike K")
    plt.ylabel("Pricing error")
    plt.title(f"Error vs Strike (sigma={sigma_fixed}, n={n_fixed}, type={option_type})")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_error_vs_vol(df, K_fixed=100, n_fixed=12, option_type="call"):
    sub = df[
        (df["K"] == K_fixed) & (df["n"] == n_fixed) & (df["option_type"] == option_type)
    ].sort_values("sigma")

    plt.figure(figsize=(8, 5))
    plt.plot(sub["sigma"], sub["cv_error"], marker="o", label="CV MC - MC")
    plt.plot(sub["sigma"], sub["tw_error"], marker="o", label="TW - MC")
    plt.plot(sub["sigma"], sub["levy_error"], marker="o", label="Levy - MC")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Volatility sigma")
    plt.ylabel("Pricing error")
    plt.title(f"Error vs Volatility (K={K_fixed}, n={n_fixed}, type={option_type})")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_runtime_bar(df):
    avg_mc = df["mc_time"].mean()
    avg_cv = df["cv_time"].mean()
    avg_tw = df["tw_time"].mean()
    avg_levy = df["levy_time"].mean()

    methods = ["MC", "CV MC", "TW", "Levy"]
    runtimes = [avg_mc, avg_cv, avg_tw, avg_levy]

    plt.figure(figsize=(8, 5))
    plt.bar(methods, runtimes)
    plt.ylabel("Average runtime (seconds)")
    plt.title("Average runtime across parameter grid")
    plt.tight_layout()
    plt.show()


def plot_levy_convergence(
    S0=100,
    K=100,
    r=0.05,
    sigma=0.2,
    T=1.0,
    n_list=None,
    n_paths=500_000,
    seed=42,
    option_type="call",
):
    """
    Show that the Levy approximation error vanishes as the number of
    monitoring dates n increases, confirming that Levy prices the
    continuous-monitoring limit while MC/TW price discrete monitoring.
    """
    if n_list is None:
        n_list = [4, 12, 26, 52, 126, 252, 504, 1000]

    # Levy price is independent of n (continuous monitoring)
    levy_price = levy_arithmetic_asian_price(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        option_type=option_type,
    )

    cv_prices = []
    tw_prices = []
    for n in n_list:
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
        )
        tw_price = turnbull_wakeman_arithmetic_asian_price(
            S0=S0,
            K=K,
            r=r,
            sigma=sigma,
            T=T,
            n=n,
            option_type=option_type,
        )
        cv_prices.append(cv_res.price)
        tw_prices.append(tw_price)

    cv_prices = np.array(cv_prices)
    tw_prices = np.array(tw_prices)

    levy_vs_cv = np.abs(levy_price - cv_prices)
    levy_vs_tw = np.abs(levy_price - tw_prices)
    tw_vs_cv = np.abs(tw_prices - cv_prices)

    # ---- Table ----
    print("\nLevy convergence as n -> inf:")
    print(f"  Levy (continuous) price = {levy_price:.6f}")
    print(
        f"  {'n':>6s}  {'CV price':>10s}  {'TW price':>10s}  {'|Levy-CV|':>10s}  {'|Levy-TW|':>10s}  {'|TW-CV|':>10s}"
    )
    print("  " + "-" * 64)
    for i, n in enumerate(n_list):
        print(
            f"  {n:6d}  {cv_prices[i]:10.6f}  {tw_prices[i]:10.6f}"
            f"  {levy_vs_cv[i]:10.6f}  {levy_vs_tw[i]:10.6f}  {tw_vs_cv[i]:10.6f}"
        )

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_list, levy_vs_cv, marker="o", label="|Levy - CV MC|")
    ax.plot(n_list, levy_vs_tw, marker="s", label="|Levy - TW|")
    ax.plot(
        n_list, tw_vs_cv, marker="^", label="|TW - CV MC|", linestyle="--", alpha=0.6
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of monitoring dates (n)")
    ax.set_ylabel("Absolute price difference")
    ax.set_title(
        "Levy continuous-monitoring error vs discrete n\n"
        f"(S0={S0}, K={K}, r={r}, σ={sigma}, T={T})"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = run_grid_experiment(
        S0=100,
        r=0.05,
        T=1.0,
        K_list=[80, 90, 100, 110, 120],
        sigma_list=[0.1, 0.2, 0.4],
        n_list=[12, 52, 252],
        n_paths=100000,
        seed=42,
        option_type="call",
    )

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print("\nFull comparison table:\n")
    print(df.round(6))

    print("\nKey summary columns:\n")
    print(
        df[
            [
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
                "cv_abs_error",
                "tw_abs_error",
                "levy_abs_error",
            ]
        ].round(6)
    )

    plot_error_vs_strike(df, sigma_fixed=0.2, n_fixed=12, option_type="call")
    plot_error_vs_vol(df, K_fixed=100, n_fixed=12, option_type="call")
    plot_runtime_bar(df)
    plot_levy_convergence()

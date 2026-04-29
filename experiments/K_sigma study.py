import os
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
    Compare MC, CV MC, TW, and Levy for one parameter set.

    Benchmark convention in this experiment:
        Plain MC is used as the baseline.

    Error definition:
        method_error = method_price - plain_mc_price

    This is a deviation-from-plain-MC study, not a true pricing error study.
    """
    result = {
        "S0": S0,
        "K": K,
        "moneyness": K / S0,
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

    eps = 1e-12
    denom = max(abs(result["mc_price"]), eps)

    result["cv_rel_error"] = result["cv_error"] / denom
    result["tw_rel_error"] = result["tw_error"] / denom
    result["levy_rel_error"] = result["levy_error"] / denom

    result["cv_abs_rel_error"] = abs(result["cv_rel_error"])
    result["tw_abs_rel_error"] = abs(result["tw_rel_error"])
    result["levy_abs_rel_error"] = abs(result["levy_rel_error"])

    return result


def run_strike_vol_grid(
    S0=100,
    r=0.05,
    T=1.0,
    n=12,
    K_list=None,
    sigma_list=None,
    n_paths=100000,
    seed=42,
    option_type="call",
):
    """
    Run a K x sigma grid with fixed monitoring frequency n.
    """
    if K_list is None:
        K_list = [70, 80, 90, 100, 110, 120, 130]

    if sigma_list is None:
        sigma_list = [0.1, 0.2, 0.3, 0.4, 0.6]

    rows = []

    total = len(K_list) * len(sigma_list)
    count = 0

    for sigma in sigma_list:
        for K in K_list:
            count += 1
            print(f"Running {count}/{total}: K={K}, sigma={sigma}, n={n}")

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


def plot_error_vs_strike(
    df,
    sigma_fixed=0.2,
    n_fixed=12,
    option_type="call",
    save_path=None,
):
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
    plt.ylabel("Method price - Plain MC price")
    plt.title(f"Deviation vs Strike (sigma={sigma_fixed}, n={n_fixed})")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_error_vs_vol(
    df,
    K_fixed=100,
    n_fixed=12,
    option_type="call",
    save_path=None,
):
    sub = df[
        (df["K"] == K_fixed)
        & (df["n"] == n_fixed)
        & (df["option_type"] == option_type)
    ].sort_values("sigma")

    plt.figure(figsize=(8, 5))
    plt.plot(sub["sigma"], sub["cv_error"], marker="o", label="CV MC - MC")
    plt.plot(sub["sigma"], sub["tw_error"], marker="o", label="TW - MC")
    plt.plot(sub["sigma"], sub["levy_error"], marker="o", label="Levy - MC")

    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Volatility sigma")
    plt.ylabel("Method price - Plain MC price")
    plt.title(f"Deviation vs Volatility (K={K_fixed}, n={n_fixed})")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_error_heatmap(
    df,
    method="tw",
    n_fixed=12,
    option_type="call",
    error_kind="signed",
):
    """
    Heatmap over (K, sigma).

    Parameters
    ----------
    method : str
        One of {"cv", "tw", "levy"}.
    error_kind : str
        "signed"       -> method_error
        "absolute"     -> method_abs_error
        "relative"     -> method_rel_error
        "abs_relative" -> method_abs_rel_error
    """
    if method not in {"cv", "tw", "levy"}:
        raise ValueError("method must be one of {'cv', 'tw', 'levy'}.")

    if error_kind == "signed":
        value_col = f"{method}_error"
        label = f"{method.upper()} signed deviation from Plain MC"
    elif error_kind == "absolute":
        value_col = f"{method}_abs_error"
        label = f"{method.upper()} absolute deviation from Plain MC"
    elif error_kind == "relative":
        value_col = f"{method}_rel_error"
        label = f"{method.upper()} relative deviation from Plain MC"
    elif error_kind == "abs_relative":
        value_col = f"{method}_abs_rel_error"
        label = f"{method.upper()} absolute relative deviation from Plain MC"
    else:
        raise ValueError(
            "error_kind must be one of {'signed', 'absolute', 'relative', 'abs_relative'}."
        )

    sub = df[
        (df["n"] == n_fixed)
        & (df["option_type"] == option_type)
    ].copy()

    pivot = sub.pivot(index="sigma", columns="K", values=value_col)

    plt.figure(figsize=(8, 5))
    plt.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
        extent=[
            pivot.columns.min(),
            pivot.columns.max(),
            pivot.index.min(),
            pivot.index.max(),
        ],
    )

    plt.colorbar(label=label)
    plt.xlabel("Strike K")
    plt.ylabel("Volatility sigma")
    plt.title(f"{label} Heatmap (n={n_fixed})")
    plt.tight_layout()
    plt.show()


def summarize_error_patterns(df, n_fixed=12, option_type="call"):
    """
    Print summary statistics that directly answer:

    1. Is error largest around ATM?
    2. Is error largest in high-vol regimes?
    3. Does Levy underprice across the full grid?
    """
    sub = df[
        (df["n"] == n_fixed)
        & (df["option_type"] == option_type)
    ].copy()

    print("\n================ Error Pattern Summary ================\n")

    for method in ["tw", "levy"]:
        err_col = f"{method}_error"
        abs_col = f"{method}_abs_error"
        rel_col = f"{method}_rel_error"
        abs_rel_col = f"{method}_abs_rel_error"

        max_abs_row = sub.loc[sub[abs_col].idxmax()]
        max_abs_rel_row = sub.loc[sub[abs_rel_col].idxmax()]

        mean_abs_by_sigma = sub.groupby("sigma")[abs_col].mean()
        mean_abs_by_K = sub.groupby("K")[abs_col].mean()
        mean_abs_rel_by_sigma = sub.groupby("sigma")[abs_rel_col].mean()
        mean_abs_rel_by_K = sub.groupby("K")[abs_rel_col].mean()

        print(f"{method.upper()} approximation:")
        print(f"  Mean signed deviation       : {sub[err_col].mean():.6f}")
        print(f"  Mean absolute deviation     : {sub[abs_col].mean():.6f}")
        print(f"  Mean relative deviation     : {sub[rel_col].mean():.6f}")
        print(f"  Mean absolute relative dev. : {sub[abs_rel_col].mean():.6f}")

        print(
            "  Max absolute deviation      : "
            f"{max_abs_row[abs_col]:.6f} at "
            f"K={max_abs_row['K']}, sigma={max_abs_row['sigma']}, "
            f"moneyness={max_abs_row['moneyness']:.2f}"
        )

        print(
            "  Max absolute relative dev.  : "
            f"{max_abs_rel_row[abs_rel_col]:.6f} at "
            f"K={max_abs_rel_row['K']}, sigma={max_abs_rel_row['sigma']}, "
            f"moneyness={max_abs_rel_row['moneyness']:.2f}"
        )

        if (sub[err_col] < 0).all():
            print("  Bias direction              : underprices Plain MC in all grid points.")
        elif (sub[err_col] > 0).all():
            print("  Bias direction              : overprices Plain MC in all grid points.")
        else:
            print("  Bias direction              : mixed signs across the grid.")

        print("\n  Mean absolute deviation by sigma:")
        print(mean_abs_by_sigma.round(6))

        print("\n  Mean absolute deviation by K:")
        print(mean_abs_by_K.round(6))

        print("\n  Mean absolute relative deviation by sigma:")
        print(mean_abs_rel_by_sigma.round(6))

        print("\n  Mean absolute relative deviation by K:")
        print(mean_abs_rel_by_K.round(6))

        print("\n--------------------------------------------------------\n")


def print_summary(df):
    cols = [
        "K",
        "moneyness",
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
        "tw_abs_error",
        "levy_abs_error",
        "tw_rel_error",
        "levy_rel_error",
        "mc_cv_plain_diff",
    ]

    print("\nStrike-volatility study summary:\n")
    print(df[cols].round(6))


if __name__ == "__main__":
    os.makedirs("../figures", exist_ok=True)

    df = run_strike_vol_grid(
        S0=100,
        r=0.05,
        T=1.0,
        n=12,
        K_list=[70, 80, 90, 100, 110, 120, 130],
        sigma_list=[0.1, 0.2, 0.3, 0.4, 0.6],
        n_paths=100000,
        seed=42,
        option_type="call",
    )

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", None)

    print_summary(df)
    summarize_error_patterns(df, n_fixed=12, option_type="call")

    # Save first two diagnostic plots
    plot_error_vs_strike(
        df,
        sigma_fixed=0.2,
        n_fixed=12,
        option_type="call",
        save_path="../figures/error_vs_strike_sigma_0p2_n12.png",
    )

    plot_error_vs_vol(
        df,
        K_fixed=100,
        n_fixed=12,
        option_type="call",
        save_path="../figures/error_vs_vol_K100_n12.png",
    )

    # Signed error heatmaps: show overpricing / underpricing regions
    plot_error_heatmap(df, method="tw", n_fixed=12, option_type="call", error_kind="signed")
    plot_error_heatmap(df, method="levy", n_fixed=12, option_type="call", error_kind="signed")

    # Absolute error heatmaps: show where error magnitude is largest
    plot_error_heatmap(df, method="tw", n_fixed=12, option_type="call", error_kind="absolute")
    plot_error_heatmap(df, method="levy", n_fixed=12, option_type="call", error_kind="absolute")

    # Relative error heatmaps: useful when price level changes sharply across K
    plot_error_heatmap(df, method="tw", n_fixed=12, option_type="call", error_kind="abs_relative")
    plot_error_heatmap(df, method="levy", n_fixed=12, option_type="call", error_kind="abs_relative")
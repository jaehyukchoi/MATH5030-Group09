"""
Greek Robustness Check for Bias-Corrected Turnbull-Wakeman
==========================================================

This script evaluates whether the bias-corrected Turnbull-Wakeman approximation
improves finite-difference Greeks across a parameter grid.

Benchmark:
    Control variate Monte Carlo finite-difference Greeks.

Methods compared:
    1. Original Turnbull-Wakeman Greeks
    2. Bias-corrected Turnbull-Wakeman Greeks

The MC Greeks use common random numbers to reduce finite-difference noise.

Theta convention:
    Theta_T is reported as dV/dT, where T is time to maturity.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

from asianoption.control_variate import arithmetic_asian_cv
from asianoption.approximation import turnbull_wakeman_arithmetic_asian_price


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"

DATA_PATH = DATA_DIR / "tw_residual_dataset_s0_grid.csv"


def train_tw_residual_model(dataset_path=DATA_PATH):
    df = pd.read_csv(dataset_path)
    df = df.copy()

    if "log_moneyness" not in df.columns:
        df["log_moneyness"] = np.log(df["K"] / df["S0"])

    if "inv_n" not in df.columns:
        df["inv_n"] = 1.0 / df["n"]

    if "inv_sqrt_n" not in df.columns:
        df["inv_sqrt_n"] = 1.0 / np.sqrt(df["n"])

    if "scaled_tw_residual" not in df.columns:
        if "tw_residual" in df.columns:
            df["scaled_tw_residual"] = df["tw_residual"] / df["S0"]
        else:
            df["scaled_tw_residual"] = (df["tw_price"] - df["cv_mc_price"]) / df["S0"]

    feature_cols = [
        "log_moneyness",
        "sigma",
        "T",
        "inv_n",
        "inv_sqrt_n",
    ]

    X = df[feature_cols]
    y = df["scaled_tw_residual"]

    model = Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=3, include_bias=False)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )

    model.fit(X, y)

    return model


def make_features(S0, K, sigma, T, n):
    return pd.DataFrame(
        {
            "log_moneyness": [np.log(K / S0)],
            "sigma": [sigma],
            "T": [T],
            "inv_n": [1.0 / n],
            "inv_sqrt_n": [1.0 / np.sqrt(n)],
        }
    )


def tw_price(S0, K, r, sigma, T, n, option_type="call"):
    return turnbull_wakeman_arithmetic_asian_price(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n=n,
        option_type=option_type,
    )


def corrected_tw_price(S0, K, r, sigma, T, n, model, option_type="call"):
    base_price = tw_price(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n=n,
        option_type=option_type,
    )

    x = make_features(
        S0=S0,
        K=K,
        sigma=sigma,
        T=T,
        n=n,
    )

    predicted_scaled_residual = model.predict(x)[0]
    predicted_residual = S0 * predicted_scaled_residual

    return base_price - predicted_residual


def cv_mc_price_with_Z(S0, K, r, sigma, T, n, Z, option_type="call"):
    res = arithmetic_asian_cv(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n=n,
        n_paths=Z.shape[0],
        seed=42,
        option_type=option_type,
        Z=Z,
    )

    return res.price


def central_difference(price_func, param_name, base_params, bump):
    params_up = base_params.copy()
    params_down = base_params.copy()

    params_up[param_name] += bump
    params_down[param_name] -= bump

    if param_name in {"S0", "sigma", "T"} and params_down[param_name] <= 0:
        raise ValueError(f"Bump makes {param_name} non-positive.")

    price_up = price_func(**params_up)
    price_down = price_func(**params_down)

    return (price_up - price_down) / (2.0 * bump)


def compute_tw_greeks(S0, K, r, sigma, T, n, option_type="call"):
    base_params = {
        "S0": S0,
        "K": K,
        "r": r,
        "sigma": sigma,
        "T": T,
        "n": n,
        "option_type": option_type,
    }

    def price_func(**params):
        return tw_price(**params)

    return {
        "Delta": central_difference(price_func, "S0", base_params, bump=0.1),
        "Vega": central_difference(price_func, "sigma", base_params, bump=0.001),
        "Rho": central_difference(price_func, "r", base_params, bump=0.0001),
        "Theta_T": central_difference(price_func, "T", base_params, bump=1.0 / 365.0),
    }


def compute_corrected_tw_greeks(S0, K, r, sigma, T, n, model, option_type="call"):
    base_params = {
        "S0": S0,
        "K": K,
        "r": r,
        "sigma": sigma,
        "T": T,
        "n": n,
        "model": model,
        "option_type": option_type,
    }

    def price_func(**params):
        return corrected_tw_price(**params)

    return {
        "Delta": central_difference(price_func, "S0", base_params, bump=0.1),
        "Vega": central_difference(price_func, "sigma", base_params, bump=0.001),
        "Rho": central_difference(price_func, "r", base_params, bump=0.0001),
        "Theta_T": central_difference(price_func, "T", base_params, bump=1.0 / 365.0),
    }


def compute_cv_mc_greeks(
    S0,
    K,
    r,
    sigma,
    T,
    n,
    n_paths=100_000,
    seed=42,
    option_type="call",
):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n_paths, n))

    base_params = {
        "S0": S0,
        "K": K,
        "r": r,
        "sigma": sigma,
        "T": T,
        "n": n,
        "Z": Z,
        "option_type": option_type,
    }

    def price_func(**params):
        return cv_mc_price_with_Z(**params)

    return {
        "Delta": central_difference(price_func, "S0", base_params, bump=0.1),
        "Vega": central_difference(price_func, "sigma", base_params, bump=0.001),
        "Rho": central_difference(price_func, "r", base_params, bump=0.0001),
        "Theta_T": central_difference(price_func, "T", base_params, bump=1.0 / 365.0),
    }


def run_greek_robustness_check(
    S0=100,
    r=0.05,
    moneyness_list=None,
    sigma_list=None,
    T_list=None,
    n_list=None,
    n_paths=100_000,
    seed=42,
    option_type="call",
    dataset_path=DATA_PATH,
):
    if moneyness_list is None:
        moneyness_list = [0.8, 0.9, 1.0, 1.1, 1.2]

    if sigma_list is None:
        sigma_list = [0.1, 0.2, 0.4]

    if T_list is None:
        T_list = [0.5, 1.0, 2.0]

    if n_list is None:
        n_list = [12, 52]

    model = train_tw_residual_model(dataset_path=dataset_path)

    rows = []

    total = len(moneyness_list) * len(sigma_list) * len(T_list) * len(n_list)
    count = 0

    for moneyness in moneyness_list:
        for sigma in sigma_list:
            for T in T_list:
                for n in n_list:
                    count += 1
                    K = S0 * moneyness

                    print(
                        f"[{count}/{total}] "
                        f"S0={S0}, K={K:.4f}, m={moneyness}, "
                        f"sigma={sigma}, T={T}, n={n}"
                    )

                    cv_mc_greeks = compute_cv_mc_greeks(
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

                    original_tw_greeks = compute_tw_greeks(
                        S0=S0,
                        K=K,
                        r=r,
                        sigma=sigma,
                        T=T,
                        n=n,
                        option_type=option_type,
                    )

                    corrected_tw_greeks = compute_corrected_tw_greeks(
                        S0=S0,
                        K=K,
                        r=r,
                        sigma=sigma,
                        T=T,
                        n=n,
                        model=model,
                        option_type=option_type,
                    )

                    for greek_name in ["Delta", "Vega", "Rho", "Theta_T"]:
                        cv_value = cv_mc_greeks[greek_name]
                        original_value = original_tw_greeks[greek_name]
                        corrected_value = corrected_tw_greeks[greek_name]

                        original_error = original_value - cv_value
                        corrected_error = corrected_value - cv_value

                        rows.append(
                            {
                                "S0": S0,
                                "K": K,
                                "moneyness": moneyness,
                                "r": r,
                                "sigma": sigma,
                                "T": T,
                                "n": n,
                                "option_type": option_type,
                                "Greek": greek_name,
                                "CV_MC": cv_value,
                                "Original_TW": original_value,
                                "Corrected_TW": corrected_value,
                                "Original_TW_Error": original_error,
                                "Corrected_TW_Error": corrected_error,
                                "Original_TW_Abs_Error": abs(original_error),
                                "Corrected_TW_Abs_Error": abs(corrected_error),
                                "Improved": abs(corrected_error) < abs(original_error),
                            }
                        )

    return pd.DataFrame(rows)


def summarize_greek_robustness(result):
    summary_rows = []

    for greek_name, sub in result.groupby("Greek"):
        original_mae = sub["Original_TW_Abs_Error"].mean()
        corrected_mae = sub["Corrected_TW_Abs_Error"].mean()

        original_rmse = np.sqrt(np.mean(sub["Original_TW_Error"] ** 2))
        corrected_rmse = np.sqrt(np.mean(sub["Corrected_TW_Error"] ** 2))

        original_max_abs_error = sub["Original_TW_Abs_Error"].max()
        corrected_max_abs_error = sub["Corrected_TW_Abs_Error"].max()

        improved_fraction = sub["Improved"].mean()

        summary_rows.append(
            {
                "Greek": greek_name,
                "Original_MAE": original_mae,
                "Corrected_MAE": corrected_mae,
                "MAE_Reduction": (
                    1.0 - corrected_mae / original_mae
                    if original_mae > 0
                    else np.nan
                ),
                "Original_RMSE": original_rmse,
                "Corrected_RMSE": corrected_rmse,
                "RMSE_Reduction": (
                    1.0 - corrected_rmse / original_rmse
                    if original_rmse > 0
                    else np.nan
                ),
                "Original_Max_Abs_Error": original_max_abs_error,
                "Corrected_Max_Abs_Error": corrected_max_abs_error,
                "Max_Error_Reduction": (
                    1.0 - corrected_max_abs_error / original_max_abs_error
                    if original_max_abs_error > 0
                    else np.nan
                ),
                "Improved_Fraction": improved_fraction,
                "Count": len(sub),
            }
        )

    summary = pd.DataFrame(summary_rows)

    return summary


def summarize_by_regime(result, group_cols):
    summary_rows = []

    for keys, sub in result.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {}
        for col, value in zip(group_cols, keys):
            row[col] = value

        for greek_name, gsub in sub.groupby("Greek"):
            original_mae = gsub["Original_TW_Abs_Error"].mean()
            corrected_mae = gsub["Corrected_TW_Abs_Error"].mean()

            row[f"{greek_name}_Original_MAE"] = original_mae
            row[f"{greek_name}_Corrected_MAE"] = corrected_mae
            row[f"{greek_name}_MAE_Reduction"] = (
                1.0 - corrected_mae / original_mae
                if original_mae > 0
                else np.nan
            )
            row[f"{greek_name}_Improved_Fraction"] = gsub["Improved"].mean()

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    result = run_greek_robustness_check(
        S0=100,
        r=0.05,
        moneyness_list=[0.8, 0.9, 1.0, 1.1, 1.2],
        sigma_list=[0.1, 0.2, 0.4],
        T_list=[0.5, 1.0, 2.0],
        n_list=[12, 52],
        n_paths=100_000,
        seed=42,
        option_type="call",
        dataset_path=DATA_PATH,
    )

    summary = summarize_greek_robustness(result)

    by_moneyness = summarize_by_regime(result, ["moneyness"])
    by_sigma = summarize_by_regime(result, ["sigma"])
    by_T = summarize_by_regime(result, ["T"])
    by_n = summarize_by_regime(result, ["n"])

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", None)

    print("\nGreek robustness summary:\n")
    print(summary.round(8))

    print("\nGreek robustness by moneyness:\n")
    print(by_moneyness.round(6))

    print("\nGreek robustness by sigma:\n")
    print(by_sigma.round(6))

    print("\nGreek robustness by T:\n")
    print(by_T.round(6))

    print("\nGreek robustness by n:\n")
    print(by_n.round(6))

    result_path = REPORT_DIR / "greeks_robustness_results.csv"
    summary_path = REPORT_DIR / "greeks_robustness_summary.csv"
    by_moneyness_path = REPORT_DIR / "greeks_robustness_by_moneyness.csv"
    by_sigma_path = REPORT_DIR / "greeks_robustness_by_sigma.csv"
    by_T_path = REPORT_DIR / "greeks_robustness_by_T.csv"
    by_n_path = REPORT_DIR / "greeks_robustness_by_n.csv"

    result.to_csv(result_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_moneyness.to_csv(by_moneyness_path, index=False)
    by_sigma.to_csv(by_sigma_path, index=False)
    by_T.to_csv(by_T_path, index=False)
    by_n.to_csv(by_n_path, index=False)

    print("\nSaved results:")
    print(result_path)
    print(summary_path)
    print(by_moneyness_path)
    print(by_sigma_path)
    print(by_T_path)
    print(by_n_path)


if __name__ == "__main__":
    main()
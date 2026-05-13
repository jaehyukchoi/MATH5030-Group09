"""
greek comparison for biased-corrected TW
=========================================

this script compares Greeks under the constant-volatitlity GBM setting

1. Control variate Monte Carlo finite-difference Greeks
2. Original Turnbull-Wakeman finite-difference Greeks
3. Bias-Corrected Turnbull-Wakeman finite-difference Greeks
"""

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.linear_model import Ridge

from asianoption.control_variate import arithmetic_asian_cv
from asianoption.approximation import turnbull_wakeman_arithmetic_asian_price

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"
DATA_PATH = DATA_DIR / "tw_residual_dataset_s0_grid.csv"
def train_tw_residual_model(dataset_path =DATA_PATH):
    df = pd.read_csv(dataset_path)
    df = df.copy()

    if "log_moneyness" not in df.columns:
        df["log_moneyness"] = np.log(df["K"] / df["S0"])
    if "inv_n" not in df.columns:
        df["inv_n"] = 1.0 / df["n"]
    feature_cols = [
        "log_moneyness",
        "sigma",
        "T",
        "inv_n",
    ]
    X = df[feature_cols]
    y = df["tw_residual"] / df["S0"]

    model = Pipeline(
        steps=[
            ("poly",PolynomialFeatures(degree=3)),
            ("scaler",StandardScaler()),
            ("ridge",Ridge(alpha=1.0)),
    ])

    model.fit(X,y)
    return model

def make_features(S0,K,sigma,T,n):
    return pd.DataFrame(
        {
            "log_moneyness":[np.log(K/S0)],
            "sigma":[sigma],
            "T":[T],
            "inv_n":[1.0 / n],
        }
    )

def tw_price(S0,K,r,sigma,T,n,option_type="call"):
    return turnbull_wakeman_arithmetic_asian_price(
        S0,K,r,sigma,T,n,option_type=option_type
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

    corrected_price = base_price - predicted_residual

    return corrected_price


def cv_mc_price_with_Z(S0, K, r, sigma, T, n, Z, option_type="call"):
    """
    Control variate Monte Carlo price using a fixed random shock matrix Z.

    This is important for finite-difference Greeks because we want to use
    common random numbers for the bumped and unbumped prices.
    """
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
    """
    Generic central finite-difference function.

    Greek approximation:

        dV/dx ≈ [V(x+h) - V(x-h)] / (2h)
    """
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
    """
    Compute finite-difference Greeks for the original TW approximation.
    """
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

    greeks = {
        "Delta": central_difference(
            price_func=price_func,
            param_name="S0",
            base_params=base_params,
            bump=0.1,
        ),
        "Vega": central_difference(
            price_func=price_func,
            param_name="sigma",
            base_params=base_params,
            bump=0.001,
        ),
        "Rho": central_difference(
            price_func=price_func,
            param_name="r",
            base_params=base_params,
            bump=0.0001,
        ),
        "Theta_T": central_difference(
            price_func=price_func,
            param_name="T",
            base_params=base_params,
            bump=1.0 / 365.0,
        ),
    }

    return greeks


def compute_corrected_tw_greeks(S0, K, r, sigma, T, n, model, option_type="call"):
    """
    Compute finite-difference Greeks for the bias-corrected TW price.
    """
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

    greeks = {
        "Delta": central_difference(
            price_func=price_func,
            param_name="S0",
            base_params=base_params,
            bump=0.1,
        ),
        "Vega": central_difference(
            price_func=price_func,
            param_name="sigma",
            base_params=base_params,
            bump=0.001,
        ),
        "Rho": central_difference(
            price_func=price_func,
            param_name="r",
            base_params=base_params,
            bump=0.0001,
        ),
        "Theta_T": central_difference(
            price_func=price_func,
            param_name="T",
            base_params=base_params,
            bump=1.0 / 365.0,
        ),
    }

    return greeks


def compute_cv_mc_greeks(
    S0,
    K,
    r,
    sigma,
    T,
    n,
    n_paths=300_000,
    seed=42,
    option_type="call",
):
    """
    Compute finite-difference Greeks using control variate Monte Carlo.

    Common random numbers are used for bumped prices to reduce Monte Carlo noise.
    """
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

    greeks = {
        "Delta": central_difference(
            price_func=price_func,
            param_name="S0",
            base_params=base_params,
            bump=0.1,
        ),
        "Vega": central_difference(
            price_func=price_func,
            param_name="sigma",
            base_params=base_params,
            bump=0.001,
        ),
        "Rho": central_difference(
            price_func=price_func,
            param_name="r",
            base_params=base_params,
            bump=0.0001,
        ),
        "Theta_T": central_difference(
            price_func=price_func,
            param_name="T",
            base_params=base_params,
            bump=1.0 / 365.0,
        ),
    }

    return greeks


def compare_greeks(
    S0=100,
    K=100,
    r=0.05,
    sigma=0.2,
    T=1.0,
    n=12,
    n_paths=300_000,
    seed=42,
    option_type="call",
    dataset_path=DATA_PATH,
):
    """
    Compare CV MC Greeks, original TW Greeks, and corrected TW Greeks.
    """
    model = train_tw_residual_model(dataset_path=dataset_path)

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

    rows = []

    for greek_name in ["Delta", "Vega", "Rho", "Theta_T"]:
        cv_value = cv_mc_greeks[greek_name]
        tw_value = original_tw_greeks[greek_name]
        corrected_value = corrected_tw_greeks[greek_name]

        rows.append(
            {
                "Greek": greek_name,
                "CV_MC": cv_value,
                "Original_TW": tw_value,
                "Corrected_TW": corrected_value,
                "Original_TW_Error": tw_value - cv_value,
                "Corrected_TW_Error": corrected_value - cv_value,
                "Original_TW_Abs_Error": abs(tw_value - cv_value),
                "Corrected_TW_Abs_Error": abs(corrected_value - cv_value),
                "Improved": abs(corrected_value - cv_value) < abs(tw_value - cv_value),
            }
        )

    result = pd.DataFrame(rows)

    return result


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    result = compare_greeks(
        S0=100,
        K=100,
        r=0.05,
        sigma=0.2,
        T=1.0,
        n=12,
        n_paths=300_000,
        seed=42,
        option_type="call",
        dataset_path=DATA_PATH,
    )

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print("\nGreek comparison results:\n")
    print(result.round(8))

    output_path = REPORT_DIR / "greeks_comparison_results.csv"
    result.to_csv(output_path, index=False)

    print("\nSaved results to:", output_path)


if __name__ == "__main__":
    main()
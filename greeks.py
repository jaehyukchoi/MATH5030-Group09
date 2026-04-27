import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from arithmetic_asian_MC import arithmetic_asian_price_mc
from control_variate import arithmetic_asian_cv
from approximation import (
    turnbull_wakeman_arithmetic_asian_price,
    levy_arithmetic_asian_price,
)


# ============================================================
# 1. Basic pricing comparison for training data
# ============================================================

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

    # Plain MC
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

    result["mc_price"] = mc_price
    result["mc_se"] = mc_se

    # Control Variate MC
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

    result["cv_price"] = cv_res.price
    result["cv_se"] = cv_res.std_error
    result["cv_beta"] = cv_res.beta
    result["variance_reduction"] = cv_res.variance_reduction

    # Turnbull-Wakeman
    tw_price = turnbull_wakeman_arithmetic_asian_price(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n=n,
        option_type=option_type,
    )

    result["tw_price"] = tw_price

    # Levy
    levy_price = levy_arithmetic_asian_price(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        option_type=option_type,
    )

    result["levy_price"] = levy_price

    # Errors against CV benchmark
    result["tw_error_vs_cv"] = result["tw_price"] - result["cv_price"]
    result["levy_error_vs_cv"] = result["levy_price"] - result["cv_price"]
    result["mc_error_vs_cv"] = result["mc_price"] - result["cv_price"]

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

    total = len(K_list) * len(sigma_list) * len(n_list)
    counter = 0

    for K in K_list:
        for sigma in sigma_list:
            for n in n_list:
                counter += 1
                print(
                    f"Running training case {counter}/{total}: "
                    f"K={K}, sigma={sigma}, T={T}, n={n}"
                )

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


# ============================================================
# 2. Feature engineering and bias model
# ============================================================

def add_engineered_features(df):
    out = df.copy()

    out["moneyness"] = out["K"] / out["S0"]
    out["log_moneyness"] = np.log(out["K"] / out["S0"])
    out["sigma_sqrt_T"] = out["sigma"] * np.sqrt(out["T"])
    out["sigma2_T"] = (out["sigma"] ** 2) * out["T"]
    out["inv_n"] = 1.0 / out["n"]
    out["inv_sqrt_n"] = 1.0 / np.sqrt(out["n"])

    return out


def make_feature_row(S0, K, r, sigma, T, n):
    row = {
        "S0": S0,
        "K": K,
        "r": r,
        "sigma": sigma,
        "T": T,
        "n": n,
        "moneyness": K / S0,
        "log_moneyness": np.log(K / S0),
        "sigma_sqrt_T": sigma * np.sqrt(T),
        "sigma2_T": sigma ** 2 * T,
        "inv_n": 1.0 / n,
        "inv_sqrt_n": 1.0 / np.sqrt(n),
    }

    return row


def train_bias_correction_model(training_df, degree=3, alpha=1.0):
    """
    Learn:

        bias = CV MC price - TW price

    Then corrected TW is:

        corrected TW = TW + predicted bias
    """

    train_df = add_engineered_features(training_df)

    feature_cols = [
        "S0",
        "K",
        "r",
        "sigma",
        "T",
        "n",
        "moneyness",
        "log_moneyness",
        "sigma_sqrt_T",
        "sigma2_T",
        "inv_n",
        "inv_sqrt_n",
    ]

    required_cols = feature_cols + ["tw_price", "cv_price"]

    missing = [c for c in required_cols if c not in train_df.columns]

    if missing:
        raise ValueError(f"Missing columns for bias model: {missing}")

    train_df = train_df.dropna(subset=required_cols).copy()

    train_df["tw_bias"] = train_df["cv_price"] - train_df["tw_price"]

    X_train = train_df[feature_cols].values
    y_train = train_df["tw_bias"].values

    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        Ridge(alpha=alpha),
    )

    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)

    print("\n===== Bias Correction Model Training Performance =====")
    print(f"Degree     : {degree}")
    print(f"Alpha      : {alpha}")
    print(f"Train MAE  : {mean_absolute_error(y_train, pred_train):.6f}")
    print(f"Train RMSE : {np.sqrt(mean_squared_error(y_train, pred_train)):.6f}")
    print(f"Train R2   : {r2_score(y_train, pred_train):.6f}")

    return model, feature_cols


# ============================================================
# 3. Price functions
# ============================================================

def tw_price_func(S0, K, r, sigma, T, n, option_type="call"):
    return turnbull_wakeman_arithmetic_asian_price(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n=n,
        option_type=option_type,
    )


def levy_price_func(S0, K, r, sigma, T, n, option_type="call"):
    return levy_arithmetic_asian_price(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        option_type=option_type,
    )


def corrected_tw_price_func(
    S0,
    K,
    r,
    sigma,
    T,
    n,
    option_type,
    bias_model,
    feature_cols,
):
    tw = tw_price_func(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n=n,
        option_type=option_type,
    )

    row = make_feature_row(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n=n,
    )

    X = np.array([[row[col] for col in feature_cols]])

    predicted_bias = bias_model.predict(X)[0]

    return tw + predicted_bias


# ============================================================
# 4. Finite difference Greeks for deterministic price functions
# ============================================================

def finite_difference_greeks(
    price_func,
    S0,
    K,
    r,
    sigma,
    T,
    n,
    option_type="call",
    h_S=None,
    h_sigma=1e-4,
    h_r=1e-4,
    h_T=1e-4,
):
    """
    Central finite difference Greeks.

    price_func signature:

        price_func(S0, K, r, sigma, T, n, option_type)

    Theta convention:

        theta = dV / dt = - dV / dT

    because T is time to maturity.
    """

    if h_S is None:
        h_S = 0.01 * S0

    V0 = price_func(S0, K, r, sigma, T, n, option_type)

    # Delta and Gamma
    V_S_up = price_func(S0 + h_S, K, r, sigma, T, n, option_type)
    V_S_down = price_func(S0 - h_S, K, r, sigma, T, n, option_type)

    delta = (V_S_up - V_S_down) / (2.0 * h_S)
    gamma = (V_S_up - 2.0 * V0 + V_S_down) / (h_S ** 2)

    # Vega
    if sigma - h_sigma <= 0:
        V_sigma_up = price_func(S0, K, r, sigma + h_sigma, T, n, option_type)
        vega = (V_sigma_up - V0) / h_sigma
    else:
        V_sigma_up = price_func(S0, K, r, sigma + h_sigma, T, n, option_type)
        V_sigma_down = price_func(S0, K, r, sigma - h_sigma, T, n, option_type)
        vega = (V_sigma_up - V_sigma_down) / (2.0 * h_sigma)

    # Rho
    V_r_up = price_func(S0, K, r + h_r, sigma, T, n, option_type)
    V_r_down = price_func(S0, K, r - h_r, sigma, T, n, option_type)
    rho = (V_r_up - V_r_down) / (2.0 * h_r)

    # Theta
    if T - h_T <= 0:
        V_T_up = price_func(S0, K, r, sigma, T + h_T, n, option_type)
        dV_dT = (V_T_up - V0) / h_T
    else:
        V_T_up = price_func(S0, K, r, sigma, T + h_T, n, option_type)
        V_T_down = price_func(S0, K, r, sigma, T - h_T, n, option_type)
        dV_dT = (V_T_up - V_T_down) / (2.0 * h_T)

    theta = -dV_dT

    return {
        "price": V0,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "rho": rho,
        "theta": theta,
    }


# ============================================================
# 5. Finite difference Greeks for CV MC benchmark
# ============================================================

def cv_price_with_Z(
    S0,
    K,
    r,
    sigma,
    T,
    n,
    option_type,
    Z,
    seed=42,
):
    n_paths = Z.shape[0]

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

    return cv_res.price


def cv_finite_difference_greeks(
    S0,
    K,
    r,
    sigma,
    T,
    n,
    option_type="call",
    n_paths=200000,
    seed=123,
    h_S=None,
    h_sigma=1e-4,
    h_r=1e-4,
    h_T=1e-4,
):
    """
    CV MC finite difference Greeks using common random numbers.

    Important:
        The same Z is reused for up/down perturbations.
        This reduces noise in finite difference estimates.
    """

    if h_S is None:
        h_S = 0.01 * S0

    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n_paths, n))

    V0 = cv_price_with_Z(S0, K, r, sigma, T, n, option_type, Z, seed)

    # Delta and Gamma
    V_S_up = cv_price_with_Z(S0 + h_S, K, r, sigma, T, n, option_type, Z, seed)
    V_S_down = cv_price_with_Z(S0 - h_S, K, r, sigma, T, n, option_type, Z, seed)

    delta = (V_S_up - V_S_down) / (2.0 * h_S)
    gamma = (V_S_up - 2.0 * V0 + V_S_down) / (h_S ** 2)

    # Vega
    if sigma - h_sigma <= 0:
        V_sigma_up = cv_price_with_Z(S0, K, r, sigma + h_sigma, T, n, option_type, Z, seed)
        vega = (V_sigma_up - V0) / h_sigma
    else:
        V_sigma_up = cv_price_with_Z(S0, K, r, sigma + h_sigma, T, n, option_type, Z, seed)
        V_sigma_down = cv_price_with_Z(S0, K, r, sigma - h_sigma, T, n, option_type, Z, seed)
        vega = (V_sigma_up - V_sigma_down) / (2.0 * h_sigma)

    # Rho
    V_r_up = cv_price_with_Z(S0, K, r + h_r, sigma, T, n, option_type, Z, seed)
    V_r_down = cv_price_with_Z(S0, K, r - h_r, sigma, T, n, option_type, Z, seed)
    rho = (V_r_up - V_r_down) / (2.0 * h_r)

    # Theta
    if T - h_T <= 0:
        V_T_up = cv_price_with_Z(S0, K, r, sigma, T + h_T, n, option_type, Z, seed)
        dV_dT = (V_T_up - V0) / h_T
    else:
        V_T_up = cv_price_with_Z(S0, K, r, sigma, T + h_T, n, option_type, Z, seed)
        V_T_down = cv_price_with_Z(S0, K, r, sigma, T - h_T, n, option_type, Z, seed)
        dV_dT = (V_T_up - V_T_down) / (2.0 * h_T)

    theta = -dV_dT

    return {
        "price": V0,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "rho": rho,
        "theta": theta,
    }


# ============================================================
# 6. Compute Greeks for one case
# ============================================================

def compute_greeks_one_case(
    S0,
    K,
    r,
    sigma,
    T,
    n,
    option_type,
    bias_model,
    feature_cols,
    n_paths_cv_greeks=200000,
):
    # TW Greeks
    tw_greeks = finite_difference_greeks(
        price_func=tw_price_func,
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n=n,
        option_type=option_type,
    )

    # Levy Greeks
    levy_greeks = finite_difference_greeks(
        price_func=levy_price_func,
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n=n,
        option_type=option_type,
    )

    # Corrected TW Greeks
    def corrected_func(S0_, K_, r_, sigma_, T_, n_, option_type_):
        return corrected_tw_price_func(
            S0=S0_,
            K=K_,
            r=r_,
            sigma=sigma_,
            T=T_,
            n=n_,
            option_type=option_type_,
            bias_model=bias_model,
            feature_cols=feature_cols,
        )

    corrected_greeks = finite_difference_greeks(
        price_func=corrected_func,
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n=n,
        option_type=option_type,
    )

    # CV MC benchmark Greeks
    cv_greeks = cv_finite_difference_greeks(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n=n,
        option_type=option_type,
        n_paths=n_paths_cv_greeks,
        seed=123,
    )

    rows = []

    rows.append({"model": "CV MC benchmark", **cv_greeks})
    rows.append({"model": "Turnbull-Wakeman", **tw_greeks})
    rows.append({"model": "Bias-Corrected TW", **corrected_greeks})
    rows.append({"model": "Levy", **levy_greeks})

    out = pd.DataFrame(rows)

    out.insert(0, "S0", S0)
    out.insert(1, "K", K)
    out.insert(2, "r", r)
    out.insert(3, "sigma", sigma)
    out.insert(4, "T", T)
    out.insert(5, "n", n)
    out.insert(6, "option_type", option_type)

    return out


# ============================================================
# 7. Greeks vs n
# ============================================================

def compute_greeks_vs_n(
    n_list,
    S0,
    K,
    r,
    sigma,
    T,
    option_type,
    bias_model,
    feature_cols,
    n_paths_cv_greeks=200000,
):
    all_rows = []

    total = len(n_list)

    for i, n in enumerate(n_list, start=1):
        print(f"\nComputing Greeks for n={n} ({i}/{total})")

        greek_df = compute_greeks_one_case(
            S0=S0,
            K=K,
            r=r,
            sigma=sigma,
            T=T,
            n=n,
            option_type=option_type,
            bias_model=bias_model,
            feature_cols=feature_cols,
            n_paths_cv_greeks=n_paths_cv_greeks,
        )

        all_rows.append(greek_df)

    return pd.concat(all_rows, ignore_index=True)


def plot_greek_vs_n(greek_df, greek_name):
    plt.figure(figsize=(9, 5))

    for model_name in greek_df["model"].unique():
        sub = greek_df[greek_df["model"] == model_name].sort_values("n")

        plt.plot(
            sub["n"],
            sub[greek_name],
            marker="o",
            label=model_name,
        )

    plt.xlabel("Number of monitoring dates n")
    plt.ylabel(greek_name)
    plt.title(f"{greek_name.capitalize()} vs Monitoring Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_all_greeks_vs_n(greek_df):
    for greek_name in ["delta", "gamma", "vega", "rho", "theta"]:
        plot_greek_vs_n(greek_df, greek_name)


# ============================================================
# 8. Main
# ============================================================

if __name__ == "__main__":

    # ------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------

    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    option_type = "call"

    n_list = [4, 6, 8, 12, 18, 26, 36, 52, 78, 126, 180, 252]

    # Training paths for price dataset
    # First test with 10000, then increase to 100000.
    n_paths_training = 100000

    # CV paths for Greek benchmark
    # First test with 50000, then increase to 200000 or more.
    n_paths_cv_greeks = 200000

    seed = 42

    # ------------------------------------------------------------
    # Step 1: Build training dataset for bias correction
    # ------------------------------------------------------------

    print("\n===== Building Bias-Correction Training Dataset =====")

    training_df = run_grid_experiment(
        S0=S0,
        r=r,
        T=T,
        K_list=[70, 80, 90, 100, 110, 120, 130],
        sigma_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        n_list=n_list,
        n_paths=n_paths_training,
        seed=seed,
        option_type=option_type,
    )

    training_df.to_csv("greeks_bias_training_grid.csv", index=False)
    print("\nSaved training data to greeks_bias_training_grid.csv")

    # ------------------------------------------------------------
    # Step 2: Train bias correction model
    # ------------------------------------------------------------

    bias_model, feature_cols = train_bias_correction_model(
        training_df=training_df,
        degree=3,
        alpha=1.0,
    )

    # ------------------------------------------------------------
    # Step 3: Greeks for one representative case
    # ------------------------------------------------------------

    print("\n===== Greeks for One Representative Case =====")

    one_case_greeks = compute_greeks_one_case(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n=52,
        option_type=option_type,
        bias_model=bias_model,
        feature_cols=feature_cols,
        n_paths_cv_greeks=n_paths_cv_greeks,
    )

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", None)

    print(one_case_greeks.round(6))

    one_case_greeks.to_csv("greeks_one_case.csv", index=False)
    print("\nSaved one-case Greeks to greeks_one_case.csv")

    # ------------------------------------------------------------
    # Step 4: Greeks vs monitoring frequency n
    # ------------------------------------------------------------

    print("\n===== Greeks vs Monitoring Frequency =====")

    greek_vs_n_df = compute_greeks_vs_n(
        n_list=n_list,
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        option_type=option_type,
        bias_model=bias_model,
        feature_cols=feature_cols,
        n_paths_cv_greeks=n_paths_cv_greeks,
    )

    print("\nGreeks vs n:")
    print(greek_vs_n_df.round(6))

    greek_vs_n_df.to_csv("greeks_vs_n.csv", index=False)
    print("\nSaved Greeks vs n to greeks_vs_n.csv")

    # ------------------------------------------------------------
    # Step 5: Plot Greeks
    # ------------------------------------------------------------

    plot_all_greeks_vs_n(greek_vs_n_df)
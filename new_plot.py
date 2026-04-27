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
# 1. Compare pricing methods for one case
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
    """
    Compare pricing methods for one parameter set.

    Benchmark:
        Plain MC and Control Variate MC are both computed.

    For bias correction:
        We will use CV MC as the benchmark because it is usually more stable
        than plain MC.
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
    result["cv_error_vs_mc"] = result["cv_price"] - result["mc_price"]
    result["tw_error_vs_mc"] = result["tw_price"] - result["mc_price"]
    result["levy_error_vs_mc"] = result["levy_price"] - result["mc_price"]

    result["cv_abs_error_vs_mc"] = abs(result["cv_error_vs_mc"])
    result["tw_abs_error_vs_mc"] = abs(result["tw_error_vs_mc"])
    result["levy_abs_error_vs_mc"] = abs(result["levy_error_vs_mc"])

    # ---------------- Deviations from CV MC ----------------
    result["mc_error_vs_cv"] = result["mc_price"] - result["cv_price"]
    result["tw_error_vs_cv"] = result["tw_price"] - result["cv_price"]
    result["levy_error_vs_cv"] = result["levy_price"] - result["cv_price"]

    result["mc_abs_error_vs_cv"] = abs(result["mc_error_vs_cv"])
    result["tw_abs_error_vs_cv"] = abs(result["tw_error_vs_cv"])
    result["levy_abs_error_vs_cv"] = abs(result["levy_error_vs_cv"])

    return result


# ============================================================
# 2. Run grid experiment
# ============================================================

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
                    f"Running case {counter}/{total}: "
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
# 3. Bias correction model
# ============================================================

def add_engineered_features(df):
    """
    Optional feature engineering.
    These features usually help the bias model because Asian option errors
    depend strongly on moneyness, volatility, maturity, and monitoring frequency.
    """

    out = df.copy()

    out["moneyness"] = out["K"] / out["S0"]
    out["log_moneyness"] = np.log(out["K"] / out["S0"])
    out["sigma_sqrt_T"] = out["sigma"] * np.sqrt(out["T"])
    out["sigma2_T"] = (out["sigma"] ** 2) * out["T"]
    out["inv_n"] = 1.0 / out["n"]
    out["inv_sqrt_n"] = 1.0 / np.sqrt(out["n"])

    return out


def train_bias_correction_model(training_df, degree=3, alpha=1.0):
    """
    Train polynomial Ridge model to learn:

        bias = cv_price - tw_price

    Then:

        corrected TW = original TW + predicted bias

    We use CV MC as the benchmark because it is more stable than plain MC.
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

    train_mae = mean_absolute_error(y_train, pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, pred_train))
    train_r2 = r2_score(y_train, pred_train)

    print("\n===== Bias Correction Model Training Performance =====")
    print(f"Degree    : {degree}")
    print(f"Alpha     : {alpha}")
    print(f"Train MAE : {train_mae:.6f}")
    print(f"Train RMSE: {train_rmse:.6f}")
    print(f"Train R2  : {train_r2:.6f}")

    return model, feature_cols


def add_corrected_tw(df, model, feature_cols):
    """
    Add corrected Turnbull-Wakeman price to df.

        predicted bias = model(X)
        corrected TW = original TW + predicted bias
    """

    out = add_engineered_features(df)

    X = out[feature_cols].values

    out["tw_predicted_bias"] = model.predict(X)
    out["tw_corrected_price"] = out["tw_price"] + out["tw_predicted_bias"]

    # Use CV as benchmark
    out["tw_corrected_error_vs_cv"] = out["tw_corrected_price"] - out["cv_price"]
    out["tw_corrected_abs_error_vs_cv"] = out["tw_corrected_error_vs_cv"].abs()

    # Also store error vs plain MC if you want to compare with old plots
    out["tw_corrected_error_vs_mc"] = out["tw_corrected_price"] - out["mc_price"]
    out["tw_corrected_abs_error_vs_mc"] = out["tw_corrected_error_vs_mc"].abs()

    return out


# ============================================================
# 4. Filtering
# ============================================================

def filter_case(df, K_fixed=100, sigma_fixed=0.2, option_type="call"):
    return df[
        (df["K"] == K_fixed)
        & (df["sigma"] == sigma_fixed)
        & (df["option_type"] == option_type)
    ].sort_values("n")


# ============================================================
# 5. Plotting functions
# ============================================================

def plot_price_vs_n(df, K_fixed=100, sigma_fixed=0.2, option_type="call"):
    sub = filter_case(df, K_fixed, sigma_fixed, option_type)

    plt.figure(figsize=(9, 5))

    plt.plot(sub["n"], sub["mc_price"], marker="o", label="Plain MC")
    plt.plot(sub["n"], sub["cv_price"], marker="o", label="CV MC")
    plt.plot(sub["n"], sub["tw_price"], marker="o", label="Turnbull-Wakeman")

    if "tw_corrected_price" in sub.columns:
        plt.plot(
            sub["n"],
            sub["tw_corrected_price"],
            marker="o",
            label="Bias-Corrected TW",
        )

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


def plot_error_vs_n_cv_benchmark(df, K_fixed=100, sigma_fixed=0.2, option_type="call"):
    """
    Error plot using CV MC as benchmark.
    This is recommended because CV MC has lower variance than plain MC.
    """

    sub = filter_case(df, K_fixed, sigma_fixed, option_type)

    plt.figure(figsize=(9, 5))

    plt.plot(
        sub["n"],
        sub["mc_price"] - sub["cv_price"],
        marker="o",
        label="Plain MC - CV MC",
    )

    plt.plot(
        sub["n"],
        sub["tw_price"] - sub["cv_price"],
        marker="o",
        label="TW - CV MC",
    )

    if "tw_corrected_error_vs_cv" in sub.columns:
        plt.plot(
            sub["n"],
            sub["tw_corrected_error_vs_cv"],
            marker="o",
            label="Bias-Corrected TW - CV MC",
        )

    plt.plot(
        sub["n"],
        sub["levy_price"] - sub["cv_price"],
        marker="o",
        label="Levy - CV MC",
    )

    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Number of monitoring dates n")
    plt.ylabel("Method price - CV MC price")
    plt.title(
        f"Deviation vs Monitoring Frequency "
        f"(K={K_fixed}, sigma={sigma_fixed}, type={option_type})"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_error_vs_n_mc_benchmark(df, K_fixed=100, sigma_fixed=0.2, option_type="call"):
    """
    Error plot using Plain MC as benchmark.
    This matches your original plot style.
    """

    sub = filter_case(df, K_fixed, sigma_fixed, option_type)

    plt.figure(figsize=(9, 5))

    plt.plot(
        sub["n"],
        sub["cv_price"] - sub["mc_price"],
        marker="o",
        label="CV MC - Plain MC",
    )

    plt.plot(
        sub["n"],
        sub["tw_price"] - sub["mc_price"],
        marker="o",
        label="TW - Plain MC",
    )

    if "tw_corrected_error_vs_mc" in sub.columns:
        plt.plot(
            sub["n"],
            sub["tw_corrected_error_vs_mc"],
            marker="o",
            label="Bias-Corrected TW - Plain MC",
        )

    plt.plot(
        sub["n"],
        sub["levy_price"] - sub["mc_price"],
        marker="o",
        label="Levy - Plain MC",
    )

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

    plt.figure(figsize=(9, 5))

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


def plot_tw_vs_corrected_scatter(df):
    """
    Scatter plot:
        CV MC benchmark vs original TW and corrected TW.

    Points closer to the 45-degree line are better.
    """

    plt.figure(figsize=(7, 7))

    plt.scatter(
        df["cv_price"],
        df["tw_price"],
        alpha=0.6,
        label="Original TW",
    )

    plt.scatter(
        df["cv_price"],
        df["tw_corrected_price"],
        alpha=0.6,
        label="Bias-Corrected TW",
    )

    min_price = min(
        df["cv_price"].min(),
        df["tw_price"].min(),
        df["tw_corrected_price"].min(),
    )

    max_price = max(
        df["cv_price"].max(),
        df["tw_price"].max(),
        df["tw_corrected_price"].max(),
    )

    plt.plot(
        [min_price, max_price],
        [min_price, max_price],
        linestyle="--",
        linewidth=1,
        label="Perfect fit",
    )

    plt.xlabel("CV MC benchmark price")
    plt.ylabel("Model price")
    plt.title("Benchmark vs Original and Bias-Corrected TW Prices")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 6. Summary
# ============================================================

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
        "tw_predicted_bias",
        "tw_corrected_price",
        "levy_price",
        "tw_error_vs_cv",
        "tw_corrected_error_vs_cv",
        "levy_error_vs_cv",
        "mc_error_vs_cv",
        "mc_cv_plain_diff",
    ]

    existing_cols = [c for c in cols if c in df.columns]

    print("\nMonitoring frequency study summary:\n")
    print(df[existing_cols].round(6))


def print_correction_quality(df):
    """
    Print original TW vs corrected TW quality using CV MC as benchmark.
    """

    original_error = df["tw_price"] - df["cv_price"]
    corrected_error = df["tw_corrected_price"] - df["cv_price"]

    original_mae = np.mean(np.abs(original_error))
    corrected_mae = np.mean(np.abs(corrected_error))

    original_rmse = np.sqrt(np.mean(original_error ** 2))
    corrected_rmse = np.sqrt(np.mean(corrected_error ** 2))

    original_max = np.max(np.abs(original_error))
    corrected_max = np.max(np.abs(corrected_error))

    improved_fraction = np.mean(np.abs(corrected_error) < np.abs(original_error))

    print("\n===== Correction Quality on Plot Dataset, CV Benchmark =====")
    print(f"Original TW MAE            : {original_mae:.6f}")
    print(f"Corrected TW MAE           : {corrected_mae:.6f}")
    print(f"MAE reduction              : {1 - corrected_mae / original_mae:.2%}")
    print(f"Original TW RMSE           : {original_rmse:.6f}")
    print(f"Corrected TW RMSE          : {corrected_rmse:.6f}")
    print(f"RMSE reduction             : {1 - corrected_rmse / original_rmse:.2%}")
    print(f"Original TW Max Abs Error  : {original_max:.6f}")
    print(f"Corrected TW Max Abs Error : {corrected_max:.6f}")
    print(f"Max error reduction        : {1 - corrected_max / original_max:.2%}")
    print(f"Improved cases             : {improved_fraction:.2%}")


# ============================================================
# 7. Main
# ============================================================

if __name__ == "__main__":

    # ------------------------------------------------------------
    # You can use small_n_paths first for testing.
    # After the script works, change to 100000.
    # ------------------------------------------------------------

    n_paths = 100000
    seed = 42

    n_list = [4, 6, 8, 12, 18, 26, 36, 52, 78, 126, 180, 252]

    # ------------------------------------------------------------
    # Step 1: Build a wider training dataset for the bias model
    # ------------------------------------------------------------

    print("\n===== Building Bias-Correction Training Dataset =====")

    training_df = run_grid_experiment(
        S0=100,
        r=0.05,
        T=1.0,
        K_list=[70, 80, 90, 100, 110, 120, 130],
        sigma_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        n_list=n_list,
        n_paths=n_paths,
        seed=seed,
        option_type="call",
    )

    training_df.to_csv("bias_correction_training_grid.csv", index=False)
    print("\nSaved training grid to bias_correction_training_grid.csv")

    # ------------------------------------------------------------
    # Step 2: Train cubic bias-correction model
    # ------------------------------------------------------------

    bias_model, feature_cols = train_bias_correction_model(
        training_df=training_df,
        degree=3,
        alpha=1.0,
    )

    # ------------------------------------------------------------
    # Step 3: Run the specific monitoring-frequency study to plot
    # ------------------------------------------------------------

    print("\n===== Running Monitoring-Frequency Plot Dataset =====")

    plot_df = run_grid_experiment(
        S0=100,
        r=0.05,
        T=1.0,
        K_list=[100],
        sigma_list=[0.2],
        n_list=n_list,
        n_paths=n_paths,
        seed=seed,
        option_type="call",
    )

    # ------------------------------------------------------------
    # Step 4: Add bias-corrected TW
    # ------------------------------------------------------------

    plot_df = add_corrected_tw(
        df=plot_df,
        model=bias_model,
        feature_cols=feature_cols,
    )

    plot_df.to_csv("monitoring_frequency_with_corrected_tw.csv", index=False)
    print("\nSaved plot dataset to monitoring_frequency_with_corrected_tw.csv")

    # ------------------------------------------------------------
    # Step 5: Print results
    # ------------------------------------------------------------

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", None)

    print_summary(plot_df)
    print_correction_quality(plot_df)

    # ------------------------------------------------------------
    # Step 6: Plot
    # ------------------------------------------------------------

    plot_price_vs_n(
        plot_df,
        K_fixed=100,
        sigma_fixed=0.2,
        option_type="call",
    )

    plot_error_vs_n_cv_benchmark(
        plot_df,
        K_fixed=100,
        sigma_fixed=0.2,
        option_type="call",
    )

    plot_error_vs_n_mc_benchmark(
        plot_df,
        K_fixed=100,
        sigma_fixed=0.2,
        option_type="call",
    )

    plot_runtime_vs_n(
        plot_df,
        K_fixed=100,
        sigma_fixed=0.2,
        option_type="call",
    )

    plot_tw_vs_corrected_scatter(plot_df)
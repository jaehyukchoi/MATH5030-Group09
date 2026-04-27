import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    make_scorer,
)


# ============================================================
# 1. Load dataset
# ============================================================

def load_dataset(path="tw_residual_dataset.csv"):
    df = pd.read_csv(path)

    required_cols = [
        "S0",
        "K",
        "moneyness",
        "log_moneyness",
        "r",
        "T",
        "sigma",
        "n",
        "cv_mc_price",
        "tw_price",
        "tw_residual",
        "tw_abs_residual",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return df


# ============================================================
# 2. Feature engineering
# ============================================================

def add_features(df):
    df = df.copy()

    # Recompute these to make sure they are consistent
    df["moneyness"] = df["K"] / df["S0"]
    df["log_moneyness"] = np.log(df["K"] / df["S0"])

    df["sigma2"] = df["sigma"] ** 2
    df["sqrt_T"] = np.sqrt(df["T"])
    df["sigma_sqrt_T"] = df["sigma"] * np.sqrt(df["T"])
    df["sigma2_T"] = df["sigma2"] * df["T"]

    df["inv_n"] = 1.0 / df["n"]
    df["inv_sqrt_n"] = 1.0 / np.sqrt(df["n"])

    # Original residual convention:
    # tw_residual = TW - benchmark
    df["tw_residual"] = df["tw_price"] - df["cv_mc_price"]
    df["tw_abs_residual"] = np.abs(df["tw_residual"])

    # Scale-normalized residual:
    # This is important if we want robustness across different S0 levels.
    df["scaled_tw_residual"] = df["tw_residual"] / df["S0"]

    return df


def get_feature_columns(version="clean"):
    """
    Important:
    We do NOT include S0 and K directly in the default model.

    Reason:
        Under GBM, Asian option prices are homogeneous:
        V(lambda*S0, lambda*K) = lambda*V(S0, K)

    Therefore, the model should learn scaled bias as a function of:
        moneyness, volatility, maturity, monitoring frequency.

    This helps generalize across different S0 levels.
    """

    if version == "minimal":
        return [
            "log_moneyness",
            "sigma",
            "T",
            "inv_n",
        ]

    if version == "clean":
        return [
            "log_moneyness",
            "sigma",
            "T",
            "inv_n",
            "inv_sqrt_n",
        ]

    if version == "enhanced":
        return [
            "log_moneyness",
            "sigma",
            "T",
            "sigma_sqrt_T",
            "inv_n",
            "inv_sqrt_n",
        ]

    if version == "old_style":
        return [
            "moneyness",
            "log_moneyness",
            "sigma",
            "sigma2",
            "T",
            "sqrt_T",
            "sigma_sqrt_T",
            "sigma2_T",
            "n",
            "inv_n",
            "inv_sqrt_n",
        ]

    raise ValueError(f"Unknown feature version: {version}")


# ============================================================
# 3. Model
# ============================================================

def build_bias_model(degree=3, alpha=1.0):
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        Ridge(alpha=alpha),
    )
    return model


# ============================================================
# 4. Evaluation helpers
# ============================================================

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def apply_correction(df, predicted_scaled_bias):
    """
    Model predicts:

        predicted_scaled_bias ≈ (TW - benchmark) / S0

    Therefore:

        predicted_price_bias = S0 * predicted_scaled_bias

    Since:
        TW residual = TW - benchmark

    Corrected TW is:

        TW_corrected = TW - predicted_price_bias
    """

    df = df.copy()

    df["predicted_scaled_tw_bias"] = predicted_scaled_bias
    df["predicted_tw_bias"] = df["S0"] * df["predicted_scaled_tw_bias"]

    df["tw_corrected_price"] = df["tw_price"] - df["predicted_tw_bias"]

    df["tw_corrected_error"] = df["tw_corrected_price"] - df["cv_mc_price"]
    df["tw_corrected_abs_error"] = np.abs(df["tw_corrected_error"])

    eps = 1e-12
    denom = np.maximum(np.abs(df["cv_mc_price"].values), eps)

    df["tw_corrected_rel_error"] = df["tw_corrected_error"] / denom
    df["tw_corrected_abs_rel_error"] = np.abs(df["tw_corrected_rel_error"])

    return df


def evaluate_correction_quality(df_eval, label="Evaluation"):
    original_mae = df_eval["tw_abs_residual"].mean()
    corrected_mae = df_eval["tw_corrected_abs_error"].mean()

    original_rmse = rmse(df_eval["tw_residual"], np.zeros(len(df_eval)))
    corrected_rmse = rmse(df_eval["tw_corrected_error"], np.zeros(len(df_eval)))

    original_max = df_eval["tw_abs_residual"].max()
    corrected_max = df_eval["tw_corrected_abs_error"].max()

    improved_fraction = (
        df_eval["tw_corrected_abs_error"] < df_eval["tw_abs_residual"]
    ).mean()

    mae_reduction = 1.0 - corrected_mae / original_mae if original_mae > 0 else np.nan
    rmse_reduction = 1.0 - corrected_rmse / original_rmse if original_rmse > 0 else np.nan
    max_reduction = 1.0 - corrected_max / original_max if original_max > 0 else np.nan

    print(f"\n===== {label}: Correction Quality =====")
    print(f"Original TW MAE             : {original_mae:.6f}")
    print(f"Corrected TW MAE            : {corrected_mae:.6f}")
    print(f"MAE reduction               : {100 * mae_reduction:.2f}%")
    print(f"Original TW RMSE            : {original_rmse:.6f}")
    print(f"Corrected TW RMSE           : {corrected_rmse:.6f}")
    print(f"RMSE reduction              : {100 * rmse_reduction:.2f}%")
    print(f"Original TW Max Abs Error   : {original_max:.6f}")
    print(f"Corrected TW Max Abs Error  : {corrected_max:.6f}")
    print(f"Max error reduction         : {100 * max_reduction:.2f}%")
    print(f"Improved cases              : {100 * improved_fraction:.2f}%")

    return {
        "label": label,
        "n_eval": len(df_eval),
        "original_mae": original_mae,
        "corrected_mae": corrected_mae,
        "mae_reduction": mae_reduction,
        "original_rmse": original_rmse,
        "corrected_rmse": corrected_rmse,
        "rmse_reduction": rmse_reduction,
        "original_max_error": original_max,
        "corrected_max_error": corrected_max,
        "max_reduction": max_reduction,
        "improved_fraction": improved_fraction,
    }


# ============================================================
# 5. Generic train/test experiment
# ============================================================

def fit_and_evaluate_split(
    df_train,
    df_test,
    label,
    degree=3,
    alpha=1.0,
    feature_version="clean",
):
    feature_cols = get_feature_columns(feature_version)

    X_train = df_train[feature_cols]
    y_train = df_train["scaled_tw_residual"]

    X_test = df_test[feature_cols]
    y_test = df_test["scaled_tw_residual"]

    model = build_bias_model(degree=degree, alpha=alpha)
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    print(f"\n===== Bias Model Performance: {label} =====")
    print(f"Feature version: {feature_version}")
    print(f"Train size: {len(df_train)}")
    print(f"Test size : {len(df_test)}")
    print(f"Train MAE scaled : {mean_absolute_error(y_train, pred_train):.8f}")
    print(f"Test MAE scaled  : {mean_absolute_error(y_test, pred_test):.8f}")
    print(f"Train RMSE scaled: {rmse(y_train, pred_train):.8f}")
    print(f"Test RMSE scaled : {rmse(y_test, pred_test):.8f}")
    print(f"Train R2         : {r2_score(y_train, pred_train):.6f}")

    if len(df_test) >= 2:
        print(f"Test R2          : {r2_score(y_test, pred_test):.6f}")
    else:
        print("Test R2          : NA")

    df_test_corrected = apply_correction(df_test, pred_test)

    quality = evaluate_correction_quality(
        df_test_corrected,
        label=label,
    )

    return {
        "name": label,
        "model": model,
        "df_test": df_test_corrected,
        "quality": quality,
    }


# ============================================================
# 6. Random train/test split
# ============================================================

def random_train_test_experiment(
    df,
    degree=3,
    alpha=1.0,
    test_size=0.25,
    feature_version="clean",
):
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        shuffle=True,
    )

    result = fit_and_evaluate_split(
        df_train=df_train,
        df_test=df_test,
        label="Random Test Set",
        degree=degree,
        alpha=alpha,
        feature_version=feature_version,
    )

    return result["model"], result["df_test"], result["quality"]


# ============================================================
# 7. K-fold cross validation
# ============================================================

def kfold_cross_validation(
    df,
    degree=3,
    alpha=1.0,
    n_splits=5,
    feature_version="clean",
):
    feature_cols = get_feature_columns(feature_version)

    X = df[feature_cols]
    y = df["scaled_tw_residual"]

    model = build_bias_model(degree=degree, alpha=alpha)

    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    scores = cross_val_score(
        model,
        X,
        y,
        scoring=mae_scorer,
        cv=cv,
    )

    mae_scores = -scores

    print("\n===== K-Fold Cross Validation on Scaled Residual =====")
    print("Fold MAE scores:", np.round(mae_scores, 8))
    print(f"CV MAE mean     : {mae_scores.mean():.8f}")
    print(f"CV MAE std      : {mae_scores.std():.8f}")

    return mae_scores


# ============================================================
# 8. Regime holdout tests
# ============================================================

def regime_holdout_experiment(
    df,
    holdout_name,
    train_mask,
    test_mask,
    degree=3,
    alpha=1.0,
    feature_version="clean",
):
    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    if len(df_train) == 0:
        print(f"\n[{holdout_name}] No training data. Skipping.")
        return None

    if len(df_test) == 0:
        print(f"\n[{holdout_name}] No test data. Skipping.")
        return None

    return fit_and_evaluate_split(
        df_train=df_train,
        df_test=df_test,
        label=f"Holdout {holdout_name}",
        degree=degree,
        alpha=alpha,
        feature_version=feature_version,
    )


def run_regime_holdout_tests(df, degree=3, alpha=1.0, feature_version="clean"):
    results = []

    # High volatility holdout
    max_sigma = df["sigma"].max()
    res = regime_holdout_experiment(
        df=df,
        holdout_name=f"High Vol sigma={max_sigma}",
        train_mask=df["sigma"] < max_sigma,
        test_mask=df["sigma"] == max_sigma,
        degree=degree,
        alpha=alpha,
        feature_version=feature_version,
    )
    if res is not None:
        results.append(res)

    # ATM / K holdout
    if 100 in set(df["K"]):
        res = regime_holdout_experiment(
            df=df,
            holdout_name="ATM K=100",
            train_mask=df["K"] != 100,
            test_mask=df["K"] == 100,
            degree=degree,
            alpha=alpha,
            feature_version=feature_version,
        )
        if res is not None:
            results.append(res)

    # High-n holdout
    max_n = df["n"].max()
    res = regime_holdout_experiment(
        df=df,
        holdout_name=f"High n={max_n}",
        train_mask=df["n"] < max_n,
        test_mask=df["n"] == max_n,
        degree=degree,
        alpha=alpha,
        feature_version=feature_version,
    )
    if res is not None:
        results.append(res)

    # Long maturity holdout
    max_T = df["T"].max()
    res = regime_holdout_experiment(
        df=df,
        holdout_name=f"Long Maturity T={max_T}",
        train_mask=df["T"] < max_T,
        test_mask=df["T"] == max_T,
        degree=degree,
        alpha=alpha,
        feature_version=feature_version,
    )
    if res is not None:
        results.append(res)

    return results


# ============================================================
# 9. K / S0 / Moneyness robustness tests
# ============================================================

def run_k_s0_moneyness_robustness_tests(
    df,
    degree=3,
    alpha=1.0,
    feature_version="clean",
):
    """
    Extra robustness tests:

    1. Leave-one-K-out:
        Tests strike robustness.

    2. Leave-one-S0-out:
        Tests spot-scale robustness.
        This only works if your dataset has multiple S0 values.

    3. Leave-one-moneyness-out:
        Tests moneyness interpolation more directly than K.

    4. Checkerboard split:
        Tests interpolation on alternating grid points.
    """

    results = []

    # ----------------------------
    # 1. Leave-one-K-out
    # ----------------------------
    unique_K = sorted(df["K"].unique())

    for K0 in unique_K:
        test_mask = df["K"] == K0
        train_mask = ~test_mask

        res = regime_holdout_experiment(
            df=df,
            holdout_name=f"Leave-one-K K={K0}",
            train_mask=train_mask,
            test_mask=test_mask,
            degree=degree,
            alpha=alpha,
            feature_version=feature_version,
        )

        if res is not None:
            results.append(res)

    # ----------------------------
    # 2. Leave-one-S0-out
    # ----------------------------
    unique_S0 = sorted(df["S0"].unique())

    if len(unique_S0) > 1:
        for S0_0 in unique_S0:
            test_mask = df["S0"] == S0_0
            train_mask = ~test_mask

            res = regime_holdout_experiment(
                df=df,
                holdout_name=f"Leave-one-S0 S0={S0_0}",
                train_mask=train_mask,
                test_mask=test_mask,
                degree=degree,
                alpha=alpha,
                feature_version=feature_version,
            )

            if res is not None:
                results.append(res)
    else:
        print("\n===== S0 Robustness Skipped =====")
        print("Only one S0 value exists in the dataset.")
        print("To test S0 robustness, regenerate dataset with S0_list such as [80, 100, 120].")

    # ----------------------------
    # 3. Leave-one-moneyness-out
    # ----------------------------
    # Floating values can be noisy, so round first.
    df_tmp = df.copy()
    df_tmp["moneyness_round"] = df_tmp["moneyness"].round(6)

    unique_m = sorted(df_tmp["moneyness_round"].unique())

    for m0 in unique_m:
        test_mask = df_tmp["moneyness_round"] == m0
        train_mask = ~test_mask

        res = regime_holdout_experiment(
            df=df_tmp,
            holdout_name=f"Leave-one-moneyness m={m0}",
            train_mask=train_mask,
            test_mask=test_mask,
            degree=degree,
            alpha=alpha,
            feature_version=feature_version,
        )

        if res is not None:
            results.append(res)

    # ----------------------------
    # 4. Checkerboard interpolation test
    # ----------------------------
    df_tmp = df.copy()

    unique_K = sorted(df_tmp["K"].unique())
    unique_sigma = sorted(df_tmp["sigma"].unique())
    unique_n = sorted(df_tmp["n"].unique())

    K_to_idx = {v: i for i, v in enumerate(unique_K)}
    sigma_to_idx = {v: i for i, v in enumerate(unique_sigma)}
    n_to_idx = {v: i for i, v in enumerate(unique_n)}

    idx_sum = (
        df_tmp["K"].map(K_to_idx)
        + df_tmp["sigma"].map(sigma_to_idx)
        + df_tmp["n"].map(n_to_idx)
    )

    test_mask = idx_sum % 2 == 0
    train_mask = ~test_mask

    res = regime_holdout_experiment(
        df=df_tmp,
        holdout_name="Checkerboard K-sigma-n interpolation",
        train_mask=train_mask,
        test_mask=test_mask,
        degree=degree,
        alpha=alpha,
        feature_version=feature_version,
    )

    if res is not None:
        results.append(res)

    return results


# ============================================================
# 10. Plots
# ============================================================

def plot_original_vs_corrected_scatter(df_eval, title="Original vs Corrected Error"):
    plt.figure(figsize=(8, 5))
    plt.scatter(
        df_eval["tw_residual"],
        df_eval["tw_corrected_error"],
        alpha=0.7,
    )
    plt.axhline(0.0, linewidth=1)
    plt.axvline(0.0, linewidth=1)
    plt.xlabel("Original TW Error")
    plt.ylabel("Corrected TW Error")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_abs_error_bar(df_eval, title="Mean Absolute Error: TW vs Corrected TW"):
    labels = ["TW", "Corrected TW"]
    values = [
        df_eval["tw_abs_residual"].mean(),
        df_eval["tw_corrected_abs_error"].mean(),
    ]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.ylabel("Mean Absolute Error")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_error_heatmap(
    df_eval,
    value_col,
    T_fixed=1.0,
    n_fixed=12,
    title=None,
):
    sub = df_eval[
        (df_eval["T"] == T_fixed)
        & (df_eval["n"] == n_fixed)
    ].copy()

    if sub.empty:
        print(f"No data for heatmap with T={T_fixed}, n={n_fixed}.")
        return

    pivot = sub.pivot_table(index="sigma", columns="K", values=value_col, aggfunc="mean")

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

    plt.colorbar(label=value_col)
    plt.xlabel("Strike K")
    plt.ylabel("Volatility sigma")

    if title is None:
        title = f"{value_col} Heatmap (T={T_fixed}, n={n_fixed})"

    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_before_after_heatmaps(df_eval, T_fixed=1.0, n_fixed=12):
    plot_error_heatmap(
        df_eval,
        value_col="tw_abs_residual",
        T_fixed=T_fixed,
        n_fixed=n_fixed,
        title=f"Original TW Absolute Error (T={T_fixed}, n={n_fixed})",
    )

    plot_error_heatmap(
        df_eval,
        value_col="tw_corrected_abs_error",
        T_fixed=T_fixed,
        n_fixed=n_fixed,
        title=f"Corrected TW Absolute Error (T={T_fixed}, n={n_fixed})",
    )


# ============================================================
# 11. Save summary
# ============================================================

def save_quality_summary(
    random_quality,
    holdout_results,
    k_s0_results,
    path="bias_correction_summary.csv",
):
    rows = [random_quality]

    for res in holdout_results:
        rows.append(res["quality"])

    for res in k_s0_results:
        rows.append(res["quality"])

    summary = pd.DataFrame(rows)
    summary.to_csv(path, index=False)

    print(f"\nSaved quality summary to {path}")
    print(summary.round(6))

    return summary


# ============================================================
# 12. Main
# ============================================================

if __name__ == "__main__":
    DATA_PATH = "tw_residual_dataset.csv"

    # Cubic was your best pricing model.
    DEGREE = 3
    ALPHA = 1.0

    # Recommended feature set:
    # "minimal", "clean", "enhanced", or "old_style"
    FEATURE_VERSION = "clean"

    df = load_dataset(DATA_PATH)
    df = add_features(df)

    print("\nDataset shape:", df.shape)
    print(df.head())

    print("\nUnique S0 values:")
    print(sorted(df["S0"].unique()))

    print("\nUnique K values:")
    print(sorted(df["K"].unique()))

    print("\nUnique moneyness values:")
    print(sorted(df["moneyness"].round(6).unique()))

    print("\nResidual description:")
    print(df["tw_residual"].describe())

    print("\nScaled residual description:")
    print(df["scaled_tw_residual"].describe())

    print("\nFeature columns:")
    print(get_feature_columns(FEATURE_VERSION))

    # ------------------------------------------------------------
    # Random train/test split
    # ------------------------------------------------------------
    model, df_test, random_quality = random_train_test_experiment(
        df,
        degree=DEGREE,
        alpha=ALPHA,
        test_size=0.25,
        feature_version=FEATURE_VERSION,
    )

    # ------------------------------------------------------------
    # K-fold CV
    # ------------------------------------------------------------
    kfold_scores = kfold_cross_validation(
        df,
        degree=DEGREE,
        alpha=ALPHA,
        n_splits=5,
        feature_version=FEATURE_VERSION,
    )

    # ------------------------------------------------------------
    # General regime holdout tests
    # ------------------------------------------------------------
    holdout_results = run_regime_holdout_tests(
        df,
        degree=DEGREE,
        alpha=ALPHA,
        feature_version=FEATURE_VERSION,
    )

    # ------------------------------------------------------------
    # K / S0 / moneyness robustness tests
    # ------------------------------------------------------------
    k_s0_results = run_k_s0_moneyness_robustness_tests(
        df,
        degree=DEGREE,
        alpha=ALPHA,
        feature_version=FEATURE_VERSION,
    )

    # ------------------------------------------------------------
    # Plots for random test set
    # ------------------------------------------------------------
    plot_original_vs_corrected_scatter(
        df_test,
        title="Random Test Set: Original vs Corrected TW Error",
    )

    plot_abs_error_bar(
        df_test,
        title="Random Test Set: TW vs Corrected TW MAE",
    )

    plot_before_after_heatmaps(
        df_test,
        T_fixed=1.0,
        n_fixed=12,
    )

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------
    df_test.to_csv("tw_bias_correction_random_test_results.csv", index=False)
    print("\nSaved random test results to tw_bias_correction_random_test_results.csv")

    save_quality_summary(
        random_quality=random_quality,
        holdout_results=holdout_results,
        k_s0_results=k_s0_results,
        path="bias_correction_summary.csv",
    )
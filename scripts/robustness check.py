import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIGURE_DIR = PROJECT_ROOT / "figures"

DATA_PATH = DATA_DIR / "tw_residual_dataset_s0_grid.csv"
RANDOM_RESULT_PATH = DATA_DIR / "tw_bias_correction_random_test_results.csv"
SUMMARY_PATH = DATA_DIR / "bias_correction_robustness_summary.csv"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "log_moneyness",
    "sigma",
    "T",
    "inv_n",
]


def load_dataset(path=DATA_PATH):
    df = pd.read_csv(path)

    required_cols = [
        "S0",
        "K",
        "T",
        "sigma",
        "n",
        "cv_mc_price",
        "tw_price",
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return df


def add_features(df):
    df = df.copy()

    df["moneyness"] = df["K"] / df["S0"]
    df["log_moneyness"] = np.log(df["moneyness"])
    df["inv_n"] = 1.0 / df["n"]

    df["tw_residual"] = df["tw_price"] - df["cv_mc_price"]
    df["tw_abs_residual"] = np.abs(df["tw_residual"])

    df["scaled_tw_residual"] = df["tw_residual"] / df["S0"]
    df["scaled_tw_abs_residual"] = np.abs(df["scaled_tw_residual"])

    return df


def build_bias_model(degree=3, alpha=1.0):
    return make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        Ridge(alpha=alpha),
    )


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def apply_correction(df, predicted_scaled_bias):
    df = df.copy()

    df["predicted_scaled_tw_bias"] = predicted_scaled_bias
    df["predicted_tw_bias"] = df["S0"] * df["predicted_scaled_tw_bias"]

    df["tw_corrected_price"] = df["tw_price"] - df["predicted_tw_bias"]
    df["tw_corrected_error"] = df["tw_corrected_price"] - df["cv_mc_price"]
    df["tw_corrected_abs_error"] = np.abs(df["tw_corrected_error"])

    df["tw_corrected_scaled_error"] = df["tw_corrected_error"] / df["S0"]
    df["tw_corrected_abs_scaled_error"] = np.abs(df["tw_corrected_scaled_error"])

    return df


def evaluate_correction_quality(df_eval, label):
    original_mae = df_eval["tw_abs_residual"].mean()
    corrected_mae = df_eval["tw_corrected_abs_error"].mean()

    original_rmse = rmse(df_eval["tw_residual"], np.zeros(len(df_eval)))
    corrected_rmse = rmse(df_eval["tw_corrected_error"], np.zeros(len(df_eval)))

    original_max = df_eval["tw_abs_residual"].max()
    corrected_max = df_eval["tw_corrected_abs_error"].max()

    original_scaled_mae = df_eval["scaled_tw_abs_residual"].mean()
    corrected_scaled_mae = df_eval["tw_corrected_abs_scaled_error"].mean()

    improved_fraction = (
        df_eval["tw_corrected_abs_error"] < df_eval["tw_abs_residual"]
    ).mean()

    mae_reduction = 1.0 - corrected_mae / original_mae if original_mae > 0 else np.nan
    rmse_reduction = 1.0 - corrected_rmse / original_rmse if original_rmse > 0 else np.nan
    max_reduction = 1.0 - corrected_max / original_max if original_max > 0 else np.nan

    scaled_mae_reduction = (
        1.0 - corrected_scaled_mae / original_scaled_mae
        if original_scaled_mae > 0
        else np.nan
    )

    print(f"\n{label}: Correction Quality")
    print(f"Original TW MAE              : {original_mae:.8f}")
    print(f"Corrected TW MAE             : {corrected_mae:.8f}")
    print(f"MAE reduction                : {100 * mae_reduction:.2f}%")
    print(f"Original TW RMSE             : {original_rmse:.8f}")
    print(f"Corrected TW RMSE            : {corrected_rmse:.8f}")
    print(f"RMSE reduction               : {100 * rmse_reduction:.2f}%")
    print(f"Original TW Max Abs Error    : {original_max:.8f}")
    print(f"Corrected TW Max Abs Error   : {corrected_max:.8f}")
    print(f"Max error reduction          : {100 * max_reduction:.2f}%")
    print(f"Original scaled MAE          : {original_scaled_mae:.10f}")
    print(f"Corrected scaled MAE         : {corrected_scaled_mae:.10f}")
    print(f"Scaled MAE reduction         : {100 * scaled_mae_reduction:.2f}%")
    print(f"Improved cases               : {100 * improved_fraction:.2f}%")

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

        "original_scaled_mae": original_scaled_mae,
        "corrected_scaled_mae": corrected_scaled_mae,
        "scaled_mae_reduction": scaled_mae_reduction,

        "improved_fraction": improved_fraction,
    }


def fit_and_evaluate_split(
    df_train,
    df_test,
    label,
    degree=3,
    alpha=1.0,
):
    if len(df_train) == 0:
        raise ValueError(f"{label}: training set is empty.")

    if len(df_test) == 0:
        raise ValueError(f"{label}: test set is empty.")

    X_train = df_train[FEATURES]
    y_train = df_train["scaled_tw_residual"]

    X_test = df_test[FEATURES]
    y_test = df_test["scaled_tw_residual"]

    model = build_bias_model(degree=degree, alpha=alpha)
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, pred_train)
    test_mae = mean_absolute_error(y_test, pred_test)

    train_rmse = rmse(y_train, pred_train)
    test_rmse = rmse(y_test, pred_test)

    train_r2 = r2_score(y_train, pred_train)
    test_r2 = r2_score(y_test, pred_test) if len(df_test) >= 2 else np.nan

    print(f"\nBias Model Performance: {label}")
    print(f"Train size        : {len(df_train)}")
    print(f"Test size         : {len(df_test)}")
    print(f"Train MAE scaled  : {train_mae:.10f}")
    print(f"Test MAE scaled   : {test_mae:.10f}")
    print(f"Train RMSE scaled : {train_rmse:.10f}")
    print(f"Test RMSE scaled  : {test_rmse:.10f}")
    print(f"Train R2          : {train_r2:.6f}")
    print(f"Test R2           : {test_r2:.6f}")

    df_test_corrected = apply_correction(df_test, pred_test)

    quality = evaluate_correction_quality(
        df_test_corrected,
        label=label,
    )

    quality["train_size"] = len(df_train)
    quality["test_size"] = len(df_test)
    quality["train_mae_scaled_model"] = train_mae
    quality["test_mae_scaled_model"] = test_mae
    quality["train_rmse_scaled_model"] = train_rmse
    quality["test_rmse_scaled_model"] = test_rmse
    quality["train_r2"] = train_r2
    quality["test_r2"] = test_r2

    return {
        "label": label,
        "model": model,
        "df_test": df_test_corrected,
        "quality": quality,
    }


def random_train_test_experiment(
    df,
    degree=3,
    alpha=1.0,
    test_size=0.25,
):
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        shuffle=True,
    )

    return fit_and_evaluate_split(
        df_train=df_train,
        df_test=df_test,
        label="Random Train/Test Split",
        degree=degree,
        alpha=alpha,
    )


def kfold_cross_validation(
    df,
    degree=3,
    alpha=1.0,
    n_splits=5,
):
    X = df[FEATURES]
    y = df["scaled_tw_residual"]

    model = build_bias_model(degree=degree, alpha=alpha)

    mae_scorer = make_scorer(
        mean_absolute_error,
        greater_is_better=False,
    )

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

    print("\nK-Fold Cross Validation on Scaled Residual")
    print("Fold MAE scores:", np.round(mae_scores, 10))
    print(f"CV MAE mean     : {mae_scores.mean():.10f}")
    print(f"CV MAE std      : {mae_scores.std():.10f}")

    return {
        "label": "K-Fold Cross Validation",
        "fold_mae_scores": mae_scores,
        "cv_mae_mean": mae_scores.mean(),
        "cv_mae_std": mae_scores.std(),
    }


def high_volatility_holdout(df, degree=3, alpha=1.0):
    max_sigma = df["sigma"].max()

    train_mask = df["sigma"] < max_sigma
    test_mask = df["sigma"] == max_sigma

    return fit_and_evaluate_split(
        df_train=df[train_mask].copy(),
        df_test=df[test_mask].copy(),
        label=f"High Volatility Holdout: sigma={max_sigma}",
        degree=degree,
        alpha=alpha,
    )


def long_maturity_holdout(df, degree=3, alpha=1.0):
    max_T = df["T"].max()

    train_mask = df["T"] < max_T
    test_mask = df["T"] == max_T

    return fit_and_evaluate_split(
        df_train=df[train_mask].copy(),
        df_test=df[test_mask].copy(),
        label=f"Long Maturity Holdout: T={max_T}",
        degree=degree,
        alpha=alpha,
    )


def high_n_holdout(df, degree=3, alpha=1.0):
    max_n = df["n"].max()

    train_mask = df["n"] < max_n
    test_mask = df["n"] == max_n

    return fit_and_evaluate_split(
        df_train=df[train_mask].copy(),
        df_test=df[test_mask].copy(),
        label=f"High-n Holdout: n={max_n}",
        degree=degree,
        alpha=alpha,
    )


def leave_one_s0_out(df, degree=3, alpha=1.0):
    results = []

    unique_s0 = sorted(df["S0"].unique())

    if len(unique_s0) <= 1:
        print("\nLeave-One-S0-Out skipped: only one S0 value exists.")
        return results

    for s0 in unique_s0:
        train_mask = df["S0"] != s0
        test_mask = df["S0"] == s0

        result = fit_and_evaluate_split(
            df_train=df[train_mask].copy(),
            df_test=df[test_mask].copy(),
            label=f"Leave-One-S0-Out: S0={s0}",
            degree=degree,
            alpha=alpha,
        )

        results.append(result)

    return results


def leave_one_moneyness_out(df, degree=3, alpha=1.0):
    results = []

    df_tmp = df.copy()
    df_tmp["moneyness_round"] = df_tmp["moneyness"].round(6)

    unique_moneyness = sorted(df_tmp["moneyness_round"].unique())

    for m in unique_moneyness:
        train_mask = df_tmp["moneyness_round"] != m
        test_mask = df_tmp["moneyness_round"] == m

        result = fit_and_evaluate_split(
            df_train=df_tmp[train_mask].copy(),
            df_test=df_tmp[test_mask].copy(),
            label=f"Leave-One-Moneyness-Out: m={m}",
            degree=degree,
            alpha=alpha,
        )

        results.append(result)

    return results


def checkerboard_interpolation_test(df, degree=3, alpha=1.0):
    df_tmp = df.copy()

    df_tmp["moneyness_round"] = df_tmp["moneyness"].round(6)

    unique_m = sorted(df_tmp["moneyness_round"].unique())
    unique_sigma = sorted(df_tmp["sigma"].unique())
    unique_T = sorted(df_tmp["T"].unique())
    unique_n = sorted(df_tmp["n"].unique())

    m_to_idx = {v: i for i, v in enumerate(unique_m)}
    sigma_to_idx = {v: i for i, v in enumerate(unique_sigma)}
    T_to_idx = {v: i for i, v in enumerate(unique_T)}
    n_to_idx = {v: i for i, v in enumerate(unique_n)}

    grid_index_sum = (
        df_tmp["moneyness_round"].map(m_to_idx)
        + df_tmp["sigma"].map(sigma_to_idx)
        + df_tmp["T"].map(T_to_idx)
        + df_tmp["n"].map(n_to_idx)
    )

    test_mask = grid_index_sum % 2 == 0
    train_mask = ~test_mask

    return fit_and_evaluate_split(
        df_train=df_tmp[train_mask].copy(),
        df_test=df_tmp[test_mask].copy(),
        label="Checkerboard Interpolation Test",
        degree=degree,
        alpha=alpha,
    )


def plot_original_vs_corrected_scatter(df_eval, title, save_path=None):
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

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    plt.show()


def plot_abs_error_bar(df_eval, title, save_path=None):
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

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    plt.show()


def plot_test_summary(summary, metric="mae_reduction", save_path=None):
    if metric not in summary.columns:
        print(f"Metric {metric} not found in summary.")
        return

    plot_df = summary.copy()
    plot_df = plot_df[plot_df[metric].notna()].copy()

    plt.figure(figsize=(12, 5))
    plt.bar(plot_df["label"], 100 * plot_df[metric])
    plt.xticks(rotation=75, ha="right")
    plt.ylabel(f"{metric} (%)")
    plt.title(f"Robustness Summary: {metric}")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    plt.show()


def flatten_results(results):
    flat = []

    for item in results:
        if item is None:
            continue

        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)

    return flat


def save_quality_summary(results, path=SUMMARY_PATH):
    flat_results = flatten_results(results)

    rows = [res["quality"] for res in flat_results]

    summary = pd.DataFrame(rows)
    summary.to_csv(path, index=False)

    print(f"\nSaved robustness summary to {path}")
    print(summary.round(6))

    return summary


if __name__ == "__main__":
    DEGREE = 3
    ALPHA = 1.0

    df = load_dataset(DATA_PATH)
    df = add_features(df)

    print("\nDataset loaded successfully.")
    print("Dataset shape:", df.shape)

    print("\nFeature columns:")
    print(FEATURES)

    print("\nUnique S0 values:")
    print(sorted(df["S0"].unique()))

    print("\nUnique moneyness values:")
    print(sorted(df["moneyness"].round(6).unique()))

    print("\nUnique sigma values:")
    print(sorted(df["sigma"].unique()))

    print("\nUnique T values:")
    print(sorted(df["T"].unique()))

    print("\nUnique n values:")
    print(sorted(df["n"].unique()))

    print("\nOriginal TW residual description:")
    print(df["tw_residual"].describe())

    print("\nScaled TW residual description:")
    print(df["scaled_tw_residual"].describe())

    random_result = random_train_test_experiment(
        df,
        degree=DEGREE,
        alpha=ALPHA,
        test_size=0.25,
    )

    kfold_result = kfold_cross_validation(
        df,
        degree=DEGREE,
        alpha=ALPHA,
        n_splits=5,
    )

    high_vol_result = high_volatility_holdout(
        df,
        degree=DEGREE,
        alpha=ALPHA,
    )

    long_T_result = long_maturity_holdout(
        df,
        degree=DEGREE,
        alpha=ALPHA,
    )

    high_n_result = high_n_holdout(
        df,
        degree=DEGREE,
        alpha=ALPHA,
    )

    s0_results = leave_one_s0_out(
        df,
        degree=DEGREE,
        alpha=ALPHA,
    )

    moneyness_results = leave_one_moneyness_out(
        df,
        degree=DEGREE,
        alpha=ALPHA,
    )

    checkerboard_result = checkerboard_interpolation_test(
        df,
        degree=DEGREE,
        alpha=ALPHA,
    )

    random_result["df_test"].to_csv(
        RANDOM_RESULT_PATH,
        index=False,
    )

    print(f"\nSaved random test details to {RANDOM_RESULT_PATH}")

    all_results = [
        random_result,
        high_vol_result,
        long_T_result,
        high_n_result,
        s0_results,
        moneyness_results,
        checkerboard_result,
    ]

    summary = save_quality_summary(
        all_results,
        path=SUMMARY_PATH,
    )

    plot_original_vs_corrected_scatter(
        random_result["df_test"],
        title="Random Test Set: Original vs Corrected TW Error",
        save_path=FIGURE_DIR / "random_original_vs_corrected_tw_error.png",
    )

    plot_abs_error_bar(
        random_result["df_test"],
        title="Random Test Set: TW vs Corrected TW MAE",
        save_path=FIGURE_DIR / "random_tw_vs_corrected_mae.png",
    )

    plot_test_summary(
        summary,
        metric="mae_reduction",
        save_path=FIGURE_DIR / "robustness_mae_reduction_summary.png",
    )

    print("\nK-fold CV summary:")
    print("Fold MAE scores:", np.round(kfold_result["fold_mae_scores"], 10))
    print("CV MAE mean:", kfold_result["cv_mae_mean"])
    print("CV MAE std :", kfold_result["cv_mae_std"])
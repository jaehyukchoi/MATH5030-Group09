"""
Stochastic-Volatility TW Residual Correction
============================================

This script trains separate residual correction models for Heston and SABR
effective-volatility Turnbull-Wakeman baselines.

For each stochastic volatility model:

    residual = TW_effective_vol_price - MC_benchmark_price

The model learns the scaled residual:

    scaled_residual = residual / S0

The corrected price is:

    corrected_price = TW_effective_vol_price - S0 * predicted_scaled_residual

This script evaluates whether the model-specific correction reduces pricing
errors relative to stochastic-volatility Monte Carlo benchmarks.
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"

def build_residual_model(degree=2, alpha=1.0):
    """
    Polynomial Ridge residual model.

    degree=2 is recommended for Heston/SABR because the feature dimension is
    larger than the original GBM residual correction problem.
    """
    return Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def evaluate_correction(df, feature_cols, model_name, degree=2, alpha=1.0, test_size=0.25, random_state=42):
    """
    Train and evaluate a residual correction model.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing tw_price, mc_price, residual, scaled_residual, and features.
    feature_cols : list[str]
        Feature columns used for residual prediction.
    model_name : str
        Name used for display.
    degree : int
        Polynomial degree.
    alpha : float
        Ridge regularization strength.
    test_size : float
        Test split fraction.
    random_state : int
        Random seed.

    Returns
    -------
    results_df : pd.DataFrame
        Prediction-level test results.
    summary : dict
        Summary metrics.
    model : sklearn Pipeline
        Trained model.
    """
    df = df.copy()

    required_cols = ["S0", "tw_price", "mc_price", "scaled_residual"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    X = df[feature_cols]
    y = df["scaled_residual"]

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    model = build_residual_model(degree=degree, alpha=alpha)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    test_df = df.loc[test_idx].copy()
    test_df["predicted_scaled_residual"] = y_pred
    test_df["predicted_residual"] = test_df["S0"] * test_df["predicted_scaled_residual"]

    test_df["corrected_price"] = test_df["tw_price"] - test_df["predicted_residual"]

    test_df["original_error"] = test_df["tw_price"] - test_df["mc_price"]
    test_df["corrected_error"] = test_df["corrected_price"] - test_df["mc_price"]

    test_df["original_abs_error"] = test_df["original_error"].abs()
    test_df["corrected_abs_error"] = test_df["corrected_error"].abs()

    original_mae = mean_absolute_error(test_df["mc_price"], test_df["tw_price"])
    corrected_mae = mean_absolute_error(test_df["mc_price"], test_df["corrected_price"])

    original_rmse = mean_squared_error(test_df["mc_price"], test_df["tw_price"]) ** 0.5
    corrected_rmse = mean_squared_error(test_df["mc_price"], test_df["corrected_price"]) ** 0.5

    original_max_abs_error = test_df["original_abs_error"].max()
    corrected_max_abs_error = test_df["corrected_abs_error"].max()

    improved_fraction = (
        test_df["corrected_abs_error"] < test_df["original_abs_error"]
    ).mean()

    residual_mae = mean_absolute_error(y_test, y_pred)
    residual_rmse = mean_squared_error(y_test, y_pred) ** 0.5
    residual_r2 = r2_score(y_test, y_pred)

    summary = {
        "model": model_name,
        "n_total": len(df),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "degree": degree,
        "alpha": alpha,
        "original_mae": original_mae,
        "corrected_mae": corrected_mae,
        "mae_reduction": 1.0 - corrected_mae / original_mae if original_mae > 0 else np.nan,
        "original_rmse": original_rmse,
        "corrected_rmse": corrected_rmse,
        "rmse_reduction": 1.0 - corrected_rmse / original_rmse if original_rmse > 0 else np.nan,
        "original_max_abs_error": original_max_abs_error,
        "corrected_max_abs_error": corrected_max_abs_error,
        "max_abs_error_reduction": (
            1.0 - corrected_max_abs_error / original_max_abs_error
            if original_max_abs_error > 0
            else np.nan
        ),
        "improved_fraction": improved_fraction,
        "residual_mae_scaled": residual_mae,
        "residual_rmse_scaled": residual_rmse,
        "residual_r2": residual_r2,
    }

    return test_df, summary, model


def run_kfold_cv(df, feature_cols, degree=2, alpha=1.0, n_splits=5, random_state=42):
    """
    K-fold CV on scaled residual prediction.
    """
    X = df[feature_cols]
    y = df["scaled_residual"]

    model = build_residual_model(degree=degree, alpha=alpha)

    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    mae_scores = -cross_val_score(
        model,
        X,
        y,
        cv=kf,
        scoring="neg_mean_absolute_error",
    )

    return mae_scores


def print_summary(summary):
    print("\n" + "=" * 80)
    print(f"{summary['model']} Residual Correction Summary")
    print("=" * 80)

    print(f"Total samples              : {summary['n_total']}")
    print(f"Train samples              : {summary['n_train']}")
    print(f"Test samples               : {summary['n_test']}")
    print(f"Polynomial degree          : {summary['degree']}")
    print(f"Ridge alpha                : {summary['alpha']}")

    print("\nPrice error metrics:")
    print(f"Original MAE               : {summary['original_mae']:.6f}")
    print(f"Corrected MAE              : {summary['corrected_mae']:.6f}")
    print(f"MAE reduction              : {summary['mae_reduction']:.2%}")

    print(f"Original RMSE              : {summary['original_rmse']:.6f}")
    print(f"Corrected RMSE             : {summary['corrected_rmse']:.6f}")
    print(f"RMSE reduction             : {summary['rmse_reduction']:.2%}")

    print(f"Original max abs error     : {summary['original_max_abs_error']:.6f}")
    print(f"Corrected max abs error    : {summary['corrected_max_abs_error']:.6f}")
    print(f"Max abs error reduction    : {summary['max_abs_error_reduction']:.2%}")

    print(f"Improved fraction          : {summary['improved_fraction']:.2%}")

    print("\nScaled residual prediction metrics:")
    print(f"Residual MAE               : {summary['residual_mae_scaled']:.8f}")
    print(f"Residual RMSE              : {summary['residual_rmse_scaled']:.8f}")
    print(f"Residual R^2               : {summary['residual_r2']:.6f}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    heston_path = DATA_DIR / "heston_tw_residual_dataset.csv"
    sabr_path = DATA_DIR / "sabr_tw_residual_dataset.csv"

    heston_df = pd.read_csv(heston_path)
    sabr_df = pd.read_csv(sabr_path)

    heston_feature_cols = [
        "log_moneyness",
        "T",
        "inv_n",
        "inv_sqrt_n",
        "v0",
        "theta",
        "kappa",
        "xi",
        "rho",
        "sigma_eff",
    ]

    sabr_feature_cols = [
        "log_moneyness",
        "T",
        "inv_n",
        "inv_sqrt_n",
        "alpha0",
        "beta",
        "nu",
        "rho",
        "sigma_eff",
    ]

    heston_test_df, heston_summary, heston_model = evaluate_correction(
        df=heston_df,
        feature_cols=heston_feature_cols,
        model_name="Heston",
        degree=2,
        alpha=1.0,
        test_size=0.25,
        random_state=42,
    )

    sabr_test_df, sabr_summary, sabr_model = evaluate_correction(
        df=sabr_df,
        feature_cols=sabr_feature_cols,
        model_name="SABR",
        degree=2,
        alpha=1.0,
        test_size=0.25,
        random_state=42,
    )

    print_summary(heston_summary)
    print_summary(sabr_summary)

    heston_cv_scores = run_kfold_cv(
        heston_df,
        heston_feature_cols,
        degree=2,
        alpha=1.0,
        n_splits=5,
        random_state=42,
    )

    sabr_cv_scores = run_kfold_cv(
        sabr_df,
        sabr_feature_cols,
        degree=2,
        alpha=1.0,
        n_splits=5,
        random_state=42,
    )

    print("\n" + "=" * 80)
    print("K-Fold CV on Scaled Residual")
    print("=" * 80)
    print("Heston fold MAE scores:", heston_cv_scores)
    print("Heston CV MAE mean:", heston_cv_scores.mean())
    print("Heston CV MAE std :", heston_cv_scores.std())

    print("\nSABR fold MAE scores:", sabr_cv_scores)
    print("SABR CV MAE mean:", sabr_cv_scores.mean())
    print("SABR CV MAE std :", sabr_cv_scores.std())

    heston_output_path = REPORT_DIR / "heston_stochastic_correction_test_results.csv"
    sabr_output_path = REPORT_DIR / "sabr_stochastic_correction_test_results.csv"
    summary_output_path = REPORT_DIR / "stochastic_bias_correction_summary.csv"

    heston_test_df.to_csv(heston_output_path, index=False)
    sabr_test_df.to_csv(sabr_output_path, index=False)

    summary_df = pd.DataFrame([heston_summary, sabr_summary])
    summary_df.to_csv(summary_output_path, index=False)

    print("\nSaved prediction-level test results:")
    print(heston_output_path)
    print(sabr_output_path)

    print("\nSaved summary:")
    print(summary_output_path)

if __name__ == "__main__":
    main()
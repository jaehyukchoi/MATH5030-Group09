import numpy as np
import pandas as pd

from pathlib import Path

from scipy.interpolate import RegularGridInterpolator

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from asianoption.control_variate import arithmetic_asian_cv
from asianoption.approximation import turnbull_wakeman_arithmetic_asian_price


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

GRID_DATA_PATH = DATA_DIR / "tw_residual_dataset_s0_grid.csv"
OFF_GRID_OUTPUT_PATH = DATA_DIR / "off_grid_interpolation_test_results.csv"



# Config


FEATURES = [
    "log_moneyness",
    "sigma",
    "T",
    "inv_n",
]



# Helpers


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def load_grid_dataset(path=GRID_DATA_PATH):
    df = pd.read_csv(path)

    df = df.copy()
    df["moneyness"] = df["K"] / df["S0"]
    df["log_moneyness"] = np.log(df["moneyness"])
    df["inv_n"] = 1.0 / df["n"]

    df["tw_residual"] = df["tw_price"] - df["cv_mc_price"]
    df["tw_abs_residual"] = np.abs(df["tw_residual"])
    df["scaled_tw_residual"] = df["tw_residual"] / df["S0"]

    return df



# Cubic interpolation model


class CubicResidualInterpolator:
    """
    Cubic interpolation model for the scaled TW residual.

    It learns:

        scaled residual = (TW - CV) / S0

    on the structured grid:

        moneyness x sigma x T x n

    For a new point, it predicts scaled residual and applies:

        corrected TW = TW - S0 * predicted_scaled_residual
    """

    def __init__(self):
        self.interpolator = None

    def fit(self, df):
        df = df.copy()

        df["moneyness"] = df["K"] / df["S0"]
        df["scaled_tw_residual"] = (df["tw_price"] - df["cv_mc_price"]) / df["S0"]

        # Average across S0 because the target is scaled by S0.
        grid_df = (
            df.groupby(["moneyness", "sigma", "T", "n"], as_index=False)
            ["scaled_tw_residual"]
            .mean()
        )

        m_grid = np.array(sorted(grid_df["moneyness"].unique()), dtype=float)
        sigma_grid = np.array(sorted(grid_df["sigma"].unique()), dtype=float)
        T_grid = np.array(sorted(grid_df["T"].unique()), dtype=float)
        n_grid = np.array(sorted(grid_df["n"].unique()), dtype=float)

        values = np.empty(
            (
                len(m_grid),
                len(sigma_grid),
                len(T_grid),
                len(n_grid),
            )
        )

        for i, m in enumerate(m_grid):
            for j, sigma in enumerate(sigma_grid):
                for k, T in enumerate(T_grid):
                    for l, n in enumerate(n_grid):
                        row = grid_df[
                            (grid_df["moneyness"] == m)
                            & (grid_df["sigma"] == sigma)
                            & (grid_df["T"] == T)
                            & (grid_df["n"] == n)
                        ]

                        if row.empty:
                            raise ValueError(
                                f"Missing grid point: m={m}, sigma={sigma}, T={T}, n={n}"
                            )

                        values[i, j, k, l] = row["scaled_tw_residual"].iloc[0]

        self.interpolator = RegularGridInterpolator(
            points=(m_grid, sigma_grid, T_grid, n_grid),
            values=values,
            method="cubic",
            bounds_error=True,
            fill_value=None,
        )

        return self

    def predict_scaled_bias(self, df):
        if self.interpolator is None:
            raise RuntimeError("Interpolator has not been fitted.")

        df = df.copy()

        points = np.column_stack(
            [
                df["K"].values / df["S0"].values,
                df["sigma"].values,
                df["T"].values,
                df["n"].values.astype(float),
            ]
        )

        return self.interpolator(points)


# Polynomial Ridge model


def build_ridge_model():
    return make_pipeline(
        PolynomialFeatures(degree=3, include_bias=False),
        StandardScaler(),
        Ridge(alpha=1.0),
    )


def fit_ridge_model(df_grid):
    X = df_grid[FEATURES]
    y = df_grid["scaled_tw_residual"]

    model = build_ridge_model()
    model.fit(X, y)

    return model



# Build off-grid test set


def build_off_grid_dataset(
    S0_list=None,
    r=0.05,
    moneyness_list=None,
    sigma_list=None,
    T_list=None,
    n_list=None,
    n_paths=200_000,
    seed=123,
    option_type="call",
):
    """
    Generate off-grid test cases.

    These parameter values are inside the original grid range,
    but they are not exactly on the original grid.

    Original grid:
        m = 0.7, 0.8, ..., 1.3
        sigma = 0.1, 0.2, ..., 0.6
        T = 0.25, 0.5, 1.0, 2.0
        n = 12, 26, 52, 126

    Off-grid examples:
        m = 0.75, 0.85, ...
        sigma = 0.15, 0.25, ...
        T = 0.375, 0.75, 1.5
        n = 18, 39, 89
    """

    if S0_list is None:
        S0_list = [90, 100, 110]

    if moneyness_list is None:
        moneyness_list = [0.75, 0.85, 0.95, 1.05, 1.15, 1.25]

    if sigma_list is None:
        sigma_list = [0.15, 0.25, 0.35, 0.45, 0.55]

    if T_list is None:
        T_list = [0.375, 0.75, 1.5]

    if n_list is None:
        n_list = [18, 39, 89]

    rows = []

    total = (
        len(S0_list)
        * len(moneyness_list)
        * len(sigma_list)
        * len(T_list)
        * len(n_list)
    )

    count = 0

    for S0 in S0_list:
        for m in moneyness_list:
            K = S0 * m

            for sigma in sigma_list:
                for T in T_list:
                    for n in n_list:
                        count += 1
                        case_seed = seed + count

                        print(
                            f"running off-grid {count}/{total}: "
                            f"S0={S0}, m={m}, K={K:.4f}, "
                            f"sigma={sigma}, T={T}, n={n}"
                        )

                        cv_result = arithmetic_asian_cv(
                            S0=S0,
                            K=K,
                            r=r,
                            T=T,
                            sigma=sigma,
                            n=n,
                            n_paths=n_paths,
                            seed=case_seed,
                            option_type=option_type,
                        )

                        cv_mc_price = cv_result.price
                        cv_mc_se = cv_result.std_error

                        tw_price = turnbull_wakeman_arithmetic_asian_price(
                            S0=S0,
                            K=K,
                            r=r,
                            sigma=sigma,
                            T=T,
                            n=n,
                            option_type=option_type,
                        )

                        tw_residual = tw_price - cv_mc_price

                        rows.append(
                            {
                                "S0": S0,
                                "K": K,
                                "moneyness": m,
                                "log_moneyness": np.log(m),
                                "r": r,
                                "T": T,
                                "sigma": sigma,
                                "n": n,
                                "inv_n": 1.0 / n,
                                "option_type": option_type,
                                "cv_mc_price": cv_mc_price,
                                "cv_mc_se": cv_mc_se,
                                "tw_price": tw_price,
                                "tw_residual": tw_residual,
                                "tw_abs_residual": abs(tw_residual),
                                "scaled_tw_residual": tw_residual / S0,
                            }
                        )

    return pd.DataFrame(rows)



# Apply corrections


def apply_cubic_correction(df_test, cubic_model):
    df = df_test.copy()

    pred_scaled = cubic_model.predict_scaled_bias(df)

    df["cubic_predicted_scaled_bias"] = pred_scaled
    df["cubic_predicted_tw_bias"] = df["S0"] * df["cubic_predicted_scaled_bias"]

    df["tw_cubic_corrected_price"] = df["tw_price"] - df["cubic_predicted_tw_bias"]
    df["tw_cubic_corrected_error"] = df["tw_cubic_corrected_price"] - df["cv_mc_price"]
    df["tw_cubic_corrected_abs_error"] = np.abs(df["tw_cubic_corrected_error"])

    return df


def apply_ridge_correction(df_test, ridge_model):
    df = df_test.copy()

    X_test = df[FEATURES]
    pred_scaled = ridge_model.predict(X_test)

    df["ridge_predicted_scaled_bias"] = pred_scaled
    df["ridge_predicted_tw_bias"] = df["S0"] * df["ridge_predicted_scaled_bias"]

    df["tw_ridge_corrected_price"] = df["tw_price"] - df["ridge_predicted_tw_bias"]
    df["tw_ridge_corrected_error"] = df["tw_ridge_corrected_price"] - df["cv_mc_price"]
    df["tw_ridge_corrected_abs_error"] = np.abs(df["tw_ridge_corrected_error"])

    return df



# Evaluation


def evaluate_methods(df):
    original_mae = df["tw_abs_residual"].mean()
    original_rmse = rmse(df["tw_residual"], np.zeros(len(df)))
    original_max = df["tw_abs_residual"].max()

    cubic_mae = df["tw_cubic_corrected_abs_error"].mean()
    cubic_rmse = rmse(df["tw_cubic_corrected_error"], np.zeros(len(df)))
    cubic_max = df["tw_cubic_corrected_abs_error"].max()
    cubic_improved = (
        df["tw_cubic_corrected_abs_error"] < df["tw_abs_residual"]
    ).mean()

    ridge_mae = df["tw_ridge_corrected_abs_error"].mean()
    ridge_rmse = rmse(df["tw_ridge_corrected_error"], np.zeros(len(df)))
    ridge_max = df["tw_ridge_corrected_abs_error"].max()
    ridge_improved = (
        df["tw_ridge_corrected_abs_error"] < df["tw_abs_residual"]
    ).mean()

    rows = [
        {
            "method": "Original TW",
            "mae": original_mae,
            "rmse": original_rmse,
            "max_abs_error": original_max,
            "mae_reduction": 0.0,
            "rmse_reduction": 0.0,
            "max_error_reduction": 0.0,
            "improved_fraction": np.nan,
        },
        {
            "method": "Polynomial Ridge Corrected TW",
            "mae": ridge_mae,
            "rmse": ridge_rmse,
            "max_abs_error": ridge_max,
            "mae_reduction": 1.0 - ridge_mae / original_mae,
            "rmse_reduction": 1.0 - ridge_rmse / original_rmse,
            "max_error_reduction": 1.0 - ridge_max / original_max,
            "improved_fraction": ridge_improved,
        },
        {
            "method": "Cubic Interpolation Corrected TW",
            "mae": cubic_mae,
            "rmse": cubic_rmse,
            "max_abs_error": cubic_max,
            "mae_reduction": 1.0 - cubic_mae / original_mae,
            "rmse_reduction": 1.0 - cubic_rmse / original_rmse,
            "max_error_reduction": 1.0 - cubic_max / original_max,
            "improved_fraction": cubic_improved,
        },
    ]

    summary = pd.DataFrame(rows)

    print("\n===== Off-Grid Interpolation Test Summary =====")
    print(summary)

    print("\nFormatted Summary:")
    for _, row in summary.iterrows():
        print(f"\nMethod: {row['method']}")
        print(f"MAE               : {row['mae']:.8f}")
        print(f"RMSE              : {row['rmse']:.8f}")
        print(f"Max Abs Error     : {row['max_abs_error']:.8f}")

        if row["method"] != "Original TW":
            print(f"MAE reduction     : {100 * row['mae_reduction']:.2f}%")
            print(f"RMSE reduction    : {100 * row['rmse_reduction']:.2f}%")
            print(f"Max reduction     : {100 * row['max_error_reduction']:.2f}%")
            print(f"Improved fraction : {100 * row['improved_fraction']:.2f}%")

    return summary


# Main


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)

    print("\nLoading original residual grid dataset...")
    df_grid = load_grid_dataset()

    print("\nGrid dataset shape:", df_grid.shape)

    print("\nFitting cubic residual interpolator...")
    cubic_model = CubicResidualInterpolator()
    cubic_model.fit(df_grid)

    print("\nFitting polynomial Ridge residual model...")
    ridge_model = fit_ridge_model(df_grid)

    print("\nGenerating off-grid test dataset...")
    df_off_grid = build_off_grid_dataset(
        S0_list=[90, 100, 110],
        r=0.05,
        moneyness_list=[0.75, 0.85, 0.95, 1.05, 1.15, 1.25],
        sigma_list=[0.15, 0.25, 0.35, 0.45, 0.55],
        T_list=[0.375, 0.75, 1.5],
        n_list=[18, 39, 89],
        n_paths=200_000,
        seed=123,
        option_type="call",
    )

    print("\nOff-grid dataset shape:", df_off_grid.shape)

    print("\nApplying cubic interpolation correction...")
    df_off_grid = apply_cubic_correction(df_off_grid, cubic_model)

    print("\nApplying polynomial Ridge correction...")
    df_off_grid = apply_ridge_correction(df_off_grid, ridge_model)

    summary = evaluate_methods(df_off_grid)

    df_off_grid.to_csv(OFF_GRID_OUTPUT_PATH, index=False)
    print(f"\nSaved off-grid test results to {OFF_GRID_OUTPUT_PATH}")

    summary_path = DATA_DIR / "off_grid_interpolation_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved off-grid summary to {summary_path}")
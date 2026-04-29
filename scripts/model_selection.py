import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"

DATA_PATH = DATA_DIR / "tw_residual_dataset_s0_grid.csv"

def load_dataset(path=DATA_PATH):
    return pd.read_csv(path)

#i think they have some economic intuition so i select these 4 parameter
def fit_tw_bias_model(df):
    features = [
        "log_moneyness",
        "sigma",
        "T",
        "inv_n",
    ]

    X = df[features]
    y = df["tw_residual"] / df["S0"]

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X,
        y,
        df,
        test_size=0.25,
        random_state=42,
    )

    model = make_pipeline(
        PolynomialFeatures(degree=3, include_bias=False),
        StandardScaler(),
        Ridge(alpha=1.0),
    )

    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    print("\n===== Bias Model Performance on Scaled Residual =====")
    print("Train MAE:", mean_absolute_error(y_train, pred_train))
    print("Test MAE :", mean_absolute_error(y_test, pred_test))
    print("Train RMSE:", np.sqrt(mean_squared_error(y_train, pred_train)))
    print("Test RMSE :", np.sqrt(mean_squared_error(y_test, pred_test)))
    print("Test R2   :", r2_score(y_test, pred_test))

    df_test = df_test.copy()

    df_test["predicted_tw_bias_scaled"] = pred_test
    df_test["predicted_tw_bias"] = df_test["S0"] * df_test["predicted_tw_bias_scaled"]

    df_test["tw_corrected_price"] = df_test["tw_price"] - df_test["predicted_tw_bias"]
    df_test["tw_corrected_error"] = df_test["tw_corrected_price"] - df_test["cv_mc_price"]
    df_test["tw_corrected_abs_error"] = np.abs(df_test["tw_corrected_error"])

    print("\nError Reduction on Test Set")
    print("Original TW MAE :", df_test["tw_abs_residual"].mean())
    print("Corrected TW MAE:", df_test["tw_corrected_abs_error"].mean())
    print("Original TW Max Abs Error :", df_test["tw_abs_residual"].max())
    print("Corrected TW Max Abs Error:", df_test["tw_corrected_abs_error"].max())

    return model, df_test


def plot_original_vs_corrected(df_test):
    plt.figure(figsize=(8, 5))
    plt.scatter(df_test["tw_residual"], df_test["tw_corrected_error"], alpha=0.7)
    plt.axhline(0.0, linewidth=1)
    plt.axvline(0.0, linewidth=1)
    plt.xlabel("Original TW Error")
    plt.ylabel("Corrected TW Error")
    plt.title("Original vs Corrected TW Error")
    plt.tight_layout()
    plt.show()


def plot_abs_error_comparison(df_test):
    labels = ["TW", "Corrected TW"]
    values = [
        df_test["tw_abs_residual"].mean(),
        df_test["tw_corrected_abs_error"].mean(),
    ]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.ylabel("Mean Absolute Error")
    plt.title("TW Bias Correction: MAE Reduction")
    plt.tight_layout()
    plt.show()


def plot_correction_heatmap(df_test, T_fixed=1.0, n_fixed=12):
    sub = df_test[
        (df_test["T"] == T_fixed)
        & (df_test["n"] == n_fixed)
    ].copy()

    if sub.empty:
        print(f"No test data for T={T_fixed}, n={n_fixed}.")
        return

    pivot = sub.pivot(index="sigma", columns="K", values="tw_corrected_abs_error")

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
    plt.colorbar(label="Corrected TW absolute error")
    plt.xlabel("Strike K")
    plt.ylabel("Volatility sigma")
    plt.title(f"Corrected TW Absolute Error Heatmap (T={T_fixed}, n={n_fixed})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = load_dataset()

    print("\nDataset shape:", df.shape)
    print(df.head())

    model, df_test = fit_tw_bias_model(df)

    plot_original_vs_corrected(df_test)
    plot_abs_error_comparison(df_test)

    df_test.to_csv("tw_bias_correction_test_results.csv", index=False)
    print("\nSaved test results to tw_bias_correction_test_results.csv")


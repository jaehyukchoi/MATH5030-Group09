"""
Build Residual Datasets for Stochastic-Volatility TW Baselines
==============================================================

This script builds residual datasets for arithmetic Asian options under
stochastic volatility models.

The purpose is to test whether an effective-volatility Turnbull-Wakeman
baseline can be corrected using model-specific residual learning.

Implemented datasets:

1. Heston residual dataset
   residual = TW_effective_vol_Heston - Heston_MC

2. SABR residual dataset
   residual = TW_effective_vol_SABR - SABR_MC

The scaled residual is defined as:

    scaled_residual = residual / S0

These datasets can later be used to train separate correction models for
Heston and SABR.
"""

import os
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
from asianoption.stochastic_volatility import (
    arithmetic_asian_heston_mc,
    arithmetic_asian_sabr_mc,
)

from asianoption.stochastic_tw import (
    heston_effective_vol,
    sabr_effective_vol,
    turnbull_wakeman_heston_effective_vol_price,
    turnbull_wakeman_sabr_effective_vol_price,
)


def build_heston_residual_dataset(
    S0_list=None,
    moneyness_list=None,
    T_list=None,
    n_list=None,
    v0_list=None,
    theta_list=None,
    kappa_list=None,
    xi_list=None,
    rho_list=None,
    r=0.05,
    n_paths=100_000,
    seed=42,
    option_type="call",
    output_path="heston_tw_residual_dataset.csv",
):
    """
    Build the Heston residual dataset.

    Residual definition:

        residual = TW_effective_vol_Heston - Heston_MC

    Scaled residual:

        scaled_residual = residual / S0
    """
    if S0_list is None:
        S0_list = [100]

    if moneyness_list is None:
        moneyness_list = [0.8, 0.9, 1.0, 1.1, 1.2]

    if T_list is None:
        T_list = [0.25, 0.5, 1.0, 2.0]

    if n_list is None:
        n_list = [12, 26, 52]

    if v0_list is None:
        v0_list = [0.01, 0.04, 0.09]

    if theta_list is None:
        theta_list = [0.01, 0.04, 0.09]

    if kappa_list is None:
        kappa_list = [0.5, 1.0, 2.0, 5.0]

    if xi_list is None:
        xi_list = [0.1, 0.3, 0.6]

    if rho_list is None:
        rho_list = [-0.8, -0.4, 0.0]

    rows = []

    grid = list(
        itertools.product(
            S0_list,
            moneyness_list,
            T_list,
            n_list,
            v0_list,
            theta_list,
            kappa_list,
            xi_list,
            rho_list,
        )
    )

    total = len(grid)

    for idx, (
        S0,
        moneyness,
        T,
        n,
        v0,
        theta,
        kappa,
        xi,
        rho,
    ) in enumerate(grid, start=1):
        K = S0 * moneyness

        print(
            f"[Heston {idx}/{total}] "
            f"S0={S0}, K={K:.4f}, m={moneyness}, T={T}, n={n}, "
            f"v0={v0}, theta={theta}, kappa={kappa}, xi={xi}, rho={rho}"
        )

        sigma_eff = heston_effective_vol(
            v0=v0,
            kappa=kappa,
            theta=theta,
            T=T,
        )

        tw_price = turnbull_wakeman_heston_effective_vol_price(
            S0=S0,
            K=K,
            r=r,
            T=T,
            v0=v0,
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho,
            n=n,
            option_type=option_type,
        )

        mc_res = arithmetic_asian_heston_mc(
            S0=S0,
            K=K,
            r=r,
            T=T,
            v0=v0,
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho,
            n=n,
            n_paths=n_paths,
            seed=seed,
            option_type=option_type,
        )

        residual = tw_price - mc_res.price
        scaled_residual = residual / S0

        row = {
            "model": "Heston",
            "S0": S0,
            "K": K,
            "moneyness": moneyness,
            "log_moneyness": np.log(K / S0),
            "r": r,
            "T": T,
            "n": n,
            "inv_n": 1.0 / n,
            "inv_sqrt_n": 1.0 / np.sqrt(n),
            "v0": v0,
            "theta": theta,
            "kappa": kappa,
            "xi": xi,
            "rho": rho,
            "sigma_eff": sigma_eff,
            "tw_price": tw_price,
            "mc_price": mc_res.price,
            "mc_std_error": mc_res.std_error,
            "residual": residual,
            "scaled_residual": scaled_residual,
            "terminal_mean": mc_res.terminal_mean,
            "average_mean": mc_res.average_mean,
            "variance_mean": mc_res.variance_mean,
            "option_type": option_type,
            "n_paths": n_paths,
            "seed": seed,
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print("\nSaved Heston residual dataset to:", output_path)
    print("Shape:", df.shape)

    return df


def build_sabr_residual_dataset(
    S0_list=None,
    moneyness_list=None,
    T_list=None,
    n_list=None,
    alpha0_list=None,
    beta_list=None,
    nu_list=None,
    rho_list=None,
    r=0.05,
    n_paths=100_000,
    seed=42,
    option_type="call",
    output_path="sabr_tw_residual_dataset.csv",
):
    """
    Build the SABR residual dataset.

    Residual definition:

        residual = TW_effective_vol_SABR - SABR_MC

    Scaled residual:

        scaled_residual = residual / S0
    """
    if S0_list is None:
        S0_list = [100]

    if moneyness_list is None:
        moneyness_list = [0.8, 0.9, 1.0, 1.1, 1.2]

    if T_list is None:
        T_list = [0.25, 0.5, 1.0, 2.0]

    if n_list is None:
        n_list = [12, 26, 52]

    if alpha0_list is None:
        alpha0_list = [0.1, 0.2, 0.3]

    if beta_list is None:
        beta_list = [0.5, 0.7, 1.0]

    if nu_list is None:
        nu_list = [0.1, 0.3, 0.6]

    if rho_list is None:
        rho_list = [-0.8, -0.4, 0.0]

    rows = []

    grid = list(
        itertools.product(
            S0_list,
            moneyness_list,
            T_list,
            n_list,
            alpha0_list,
            beta_list,
            nu_list,
            rho_list,
        )
    )

    total = len(grid)

    for idx, (
        S0,
        moneyness,
        T,
        n,
        alpha0,
        beta,
        nu,
        rho,
    ) in enumerate(grid, start=1):
        K = S0 * moneyness

        print(
            f"[SABR {idx}/{total}] "
            f"S0={S0}, K={K:.4f}, m={moneyness}, T={T}, n={n}, "
            f"alpha0={alpha0}, beta={beta}, nu={nu}, rho={rho}"
        )

        sigma_eff = sabr_effective_vol(
            S0=S0,
            alpha0=alpha0,
            beta=beta,
        )

        tw_price = turnbull_wakeman_sabr_effective_vol_price(
            S0=S0,
            K=K,
            r=r,
            T=T,
            alpha0=alpha0,
            beta=beta,
            nu=nu,
            rho=rho,
            n=n,
            option_type=option_type,
        )

        mc_res = arithmetic_asian_sabr_mc(
            F0=S0,
            K=K,
            r=r,
            T=T,
            alpha0=alpha0,
            beta=beta,
            nu=nu,
            rho=rho,
            n=n,
            n_paths=n_paths,
            seed=seed,
            option_type=option_type,
        )

        residual = tw_price - mc_res.price
        scaled_residual = residual / S0

        row = {
            "model": "SABR",
            "S0": S0,
            "K": K,
            "moneyness": moneyness,
            "log_moneyness": np.log(K / S0),
            "r": r,
            "T": T,
            "n": n,
            "inv_n": 1.0 / n,
            "inv_sqrt_n": 1.0 / np.sqrt(n),
            "alpha0": alpha0,
            "beta": beta,
            "nu": nu,
            "rho": rho,
            "sigma_eff": sigma_eff,
            "tw_price": tw_price,
            "mc_price": mc_res.price,
            "mc_std_error": mc_res.std_error,
            "residual": residual,
            "scaled_residual": scaled_residual,
            "terminal_mean": mc_res.terminal_mean,
            "average_mean": mc_res.average_mean,
            "alpha_terminal_mean": mc_res.alpha_terminal_mean,
            "option_type": option_type,
            "n_paths": n_paths,
            "seed": seed,
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print("\nSaved SABR residual dataset to:", output_path)
    print("Shape:", df.shape)

    return df


def print_dataset_summary(df, name):
    print("\n" + "=" * 80)
    print(f"{name} Dataset Summary")
    print("=" * 80)
    print("Shape:", df.shape)
    print("\nResidual summary:")
    print(df["residual"].describe())
    print("\nScaled residual summary:")
    print(df["scaled_residual"].describe())
    print("\nMean absolute residual:", df["residual"].abs().mean())
    print("Mean absolute scaled residual:", df["scaled_residual"].abs().mean())


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Smaller default grid for first run.

    heston_df = build_heston_residual_dataset(
        S0_list=[100],
        moneyness_list=[0.8, 0.9, 1.0, 1.1, 1.2],
        T_list=[0.5, 1.0, 2.0],
        n_list=[12, 26, 52],
        v0_list=[0.01, 0.04, 0.09],
        theta_list=[0.01, 0.04, 0.09],
        kappa_list=[0.5, 2.0],
        xi_list=[0.1, 0.3, 0.6],
        rho_list=[-0.8, -0.4, 0.0],
        r=0.05,
        n_paths=50_000,
        seed=42,
        option_type="call",
        output_path=DATA_DIR / "heston_tw_residual_dataset.csv",
    )

    print_dataset_summary(heston_df, "Heston")

    sabr_df = build_sabr_residual_dataset(
        S0_list=[100],
        moneyness_list=[0.8, 0.9, 1.0, 1.1, 1.2],
        T_list=[0.5, 1.0, 2.0],
        n_list=[12, 26, 52],
        alpha0_list=[0.1, 0.2, 0.3],
        beta_list=[0.5, 0.7, 1.0],
        nu_list=[0.1, 0.3, 0.6],
        rho_list=[-0.8, -0.4, 0.0],
        r=0.05,
        n_paths=50_000,
        seed=42,
        option_type="call",
        output_path=DATA_DIR / "sabr_tw_residual_dataset.csv",
    )

    print_dataset_summary(sabr_df, "SABR")
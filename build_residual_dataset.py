import numpy as np
import pandas as pd

from control_variate import arithmetic_asian_cv
from approximation import turnbull_wakeman_arithmetic_asian_price


def build_tw_residual_dataset(
    S0_list=None,
    r=0.05,
    T_list=None,
    moneyness_list=None,
    n_list=None,
    sigma_list=None,
    n_paths=200_000,
    seed=42,
    option_type="call",
):
    """
    Build TW residual dataset.
    """

    if S0_list is None:
        S0_list = [80,90,100,110,120]

    if T_list is None:
        T_list = [0.25, 0.5, 1.0, 2.0]

    if moneyness_list is None:
        moneyness_list = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

    if sigma_list is None:
        sigma_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    if n_list is None:
        n_list = [12, 26, 52, 126]

    rows = []

    total = (
        len(S0_list)
        * len(T_list)
        * len(moneyness_list)
        * len(sigma_list)
        * len(n_list)
    )

    count = 0

    for S0 in S0_list:
        for T in T_list:
            for moneyness in moneyness_list:
                K = S0 * moneyness
                for sigma in sigma_list:
                    for n in n_list:
                        count += 1

                        print(
                            f"running {count}/{total}: "
                            f"S0={S0}, K={K:.4f}, m={moneyness}, "
                            f"T={T}, sigma={sigma}, n={n}"
                        )

                        case_seed = seed + count

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

                        residual = tw_price - cv_mc_price

                        rows.append(
                            {
                                "S0": S0,
                                "K": K,
                                "moneyness": K / S0,
                                "log_moneyness": np.log(K / S0),
                                "r": r,
                                "T": T,
                                "sigma": sigma,
                                "sigma2": sigma ** 2,
                                "n": n,
                                "inv_n": 1.0 / n,
                                "inv_sqrt_n": 1.0 / np.sqrt(n),
                                "option_type": option_type,

                                "cv_mc_price": cv_mc_price,
                                "cv_mc_se": cv_mc_se,
                                "tw_price": tw_price,

                                # raw residual
                                "tw_residual": residual,
                                "tw_abs_residual": abs(residual),
                                "tw_rel_residual": residual / max(abs(cv_mc_price), 1e-12),

                                # scale-normalized residual
                                "scaled_tw_residual": residual / S0,
                                "scaled_tw_abs_residual": abs(residual / S0),
                            }
                        )

    return pd.DataFrame(rows)


if __name__ == "__main__":

    df = build_tw_residual_dataset(
        S0_list=[80,90,100,110,120],
        r=0.05,
        T_list=[0.25, 0.5, 1.0, 2.0],
        moneyness_list=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
        sigma_list=[0.1, 0.2, 0.3, 0.4,0.5, 0.6],
        n_list=[12, 26, 52, 126],
        n_paths=200_000,
        seed=42,
        option_type="call",
    )

    print("\nHead:")
    print(df.head())

    print("\nDescribe:")
    print(df.describe())

    print("\nUnique S0:")
    print(sorted(df["S0"].unique()))

    print("\nUnique moneyness:")
    print(sorted(df["moneyness"].round(6).unique()))

    print("\nDataset shape:")
    print(df.shape)

    df.to_csv("tw_residual_dataset_s0_grid.csv", index=False)

    print("\nSaved to tw_residual_dataset_s0_grid.csv")
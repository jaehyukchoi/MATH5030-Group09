import numpy as np
import pandas as pd
from arithmetic_asian_MC import arithmetic_asian_price_mc
from approximation import turnbull_wakeman_arithmetic_asian_price

import numpy as np
import pandas as pd

from arithmetic_asian_MC import arithmetic_asian_price_mc
from approximation import turnbull_wakeman_arithmetic_asian_price

def build_tw_residual_dataset(
        S0=100,
        r = 0.05,
        T_list = None,
        K_list = None,
        n_list = None,
        sigma_list = None,
        n_paths = 200000,
        seed=42,
        option_type="call",
):
    if T_list is None:
        T_list = [0.25, 0.5, 1.0, 2.0]
    if K_list is None:
        K_list = [70, 80, 90, 100, 110, 120, 130]
    if sigma_list is None:
        sigma_list = [0.1, 0.2, 0.3, 0.4, 0.6]
    if n_list is None:
        n_list = [12, 26, 52, 126, 252]

    rows = []
    total = len(T_list)*len(K_list)*len(sigma_list)*len(n_list)
    count = 0

    for T in T_list:
        for K in K_list:
            for sigma in sigma_list:
                for n in n_list:
                    count+=1
                    print(
                        f"running {count}/{total}: "
                        f"T = {T}, K={K}, sigma={sigma}, n={n}")

                    case_seed = seed + count

                    mc_price,mc_se = arithmetic_asian_price_mc(
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
                    tw_price = turnbull_wakeman_arithmetic_asian_price(
                        S0=S0,
                        K=K,
                        r=r,
                        sigma=sigma,
                        T=T,
                        n=n,
                        option_type=option_type,
                    )

                    residual = tw_price - mc_price
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
                            "option_type": option_type,
                            "mc_price": mc_price,
                            "mc_se": mc_se,
                            "tw_price": tw_price,
                            "tw_residual": residual,
                            "tw_abs_residual": abs(residual),
                            "tw_rel_residual": residual / max(abs(mc_price), 1e-12),

                        }

                    )

    return pd.DataFrame(rows)

if __name__ == "__main__":

    df = build_tw_residual_dataset(
        S0=100,
        r=0.05,
        T_list=[0.25, 0.5, 1.0, 2.0],
        K_list=[70, 80, 90, 100, 110, 120, 130],
        sigma_list=[0.1, 0.2, 0.3, 0.4, 0.6],
        n_list=[12, 26, 52, 126],
        n_paths=200_000,
        seed=42,
        option_type="call",
    )

    print(df.head())

    print(df.describe())

    df.to_csv("tw_residual_dataset.csv", index=False)

    print("\nSaved to tw_residual_dataset.csv")





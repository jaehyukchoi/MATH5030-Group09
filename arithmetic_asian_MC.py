import numpy as np
from utils import (

    simulate_gbm_paths,
    arithmetic_average_mc,
    discounted,
)

def arithmetic_asian_price_mc(S0,K,r,T,sigma,n,n_paths=100000,seed=42,option_type="call"):
    if option_type not in {"call","put"}:
        raise ValueError("optional_type must be 'call' or 'put‘")
    S_path = simulate_gbm_paths(S0=S0,r=r,sigma=sigma,n=n,n_paths=n_paths,seed=seed,T=T)
    average = arithmetic_average_mc(S_path)
    if option_type == "call":
        payoff=np.maximum(average-K,0)
    else:
        payoff=np.maximum(K-average,0.0)

    discounted_payoff = discounted(payoff,r,T)
    price = np.mean(discounted_payoff)
    std_error = np.std(discounted_payoff,ddof=1)/np.sqrt(n_paths)

    return price,std_error


if __name__ =="__main__":
    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    n = 12
    price,std = arithmetic_asian_price_mc(S0,K,r,T,sigma,n)

    print(price,std)




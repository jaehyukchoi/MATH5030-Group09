import numpy as np
from dataclasses import dataclass
import math

def arithmetic_asian_price_mc(S0,K,r,T,sigma,n,n_paths=100000,seed=42):
    rng=np.random.default_rng(seed)
    dt = T/n
    Z = rng.normal(size=(n_paths,n))

    log_return = (r-1/2*sigma**2)*dt+sigma*Z*np.sqrt(dt)
    log_S = np.cumsum(log_return,axis=1)
    S = S0*np.exp(log_S)

    #Arithemetic Average
    S_path = np.concatenate([S0*np.ones((n_paths,1)),S],axis=1)
    average_price = np.mean(S_path,axis=1)
    payoff = np.maximum(average_price-K,0)
    discounted_payoff = np.exp(-r*T)*payoff
    std_error = np.std(discounted_payoff,ddof = 1)/np.sqrt(n_paths)

    price = np.exp(-r*T)*np.mean(payoff)
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




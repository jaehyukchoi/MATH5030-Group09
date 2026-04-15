import numpy as np
import math
from dataclasses import dataclass


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class GeometricAsianResult:
    price: float
    d1: float
    d2: float
    adjusted_spot: float
    sigma_g: float
    mu_g: float


def simulate_gbm_paths(S0, r, sigma, T, n, n_paths=100000, seed=42):
    rng = np.random.default_rng(seed)
    dt = T / n
    Z = rng.normal(size=(n_paths, n))

    log_return = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_S = np.cumsum(log_return, axis=1)
    S = S0 * np.exp(log_S)

    S_path = np.concatenate([S0 * np.ones((n_paths, 1)), S], axis=1)
    return S_path


def arithmetic_average_mc(S_path):
    return np.mean(S_path, axis=1)


def geometric_average_mc(S_path):
    return np.exp(np.mean(np.log(S_path), axis=1))


def discounted(X, r, T):
    return np.exp(-r * T) * X
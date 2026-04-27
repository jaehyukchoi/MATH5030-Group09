# Bias-Corrected Turnbull-Wakeman Approximation for Arithmetic Asian Options

This project studies the pricing of arithmetic Asian options under the Black-Scholes geometric Brownian motion framework. Since arithmetic Asian options do not admit a simple closed-form solution, the project compares Monte Carlo methods, analytical approximations, and a data-driven residual correction approach.

The main goal is to build a pricing framework that keeps the speed of analytical approximations while improving their accuracy relative to a high-precision Monte Carlo benchmark.

---

## 1. Motivation

Asian options depend on the average price of the underlying asset over time. For a discretely monitored arithmetic Asian call option, the arithmetic average is

$$
A = \frac{1}{n}\sum_{i=1}^{n} S_{t_i}.
$$

The call payoff is

$$
\max(A-K,0) = (A-K)^+.
$$

Unlike geometric Asian options, arithmetic Asian options do not have a simple closed-form pricing formula. This creates a practical trade-off:

- Plain Monte Carlo is flexible but computationally expensive.
- Control variate Monte Carlo is more accurate but still simulation-based.
- Analytical approximations such as Turnbull-Wakeman are fast but can have systematic bias.

This project investigates whether the bias of a fast approximation can be learned and corrected.

---

## 2. Project Structure

```text
AsianOption/
│
├── arithmetic_asian_MC.py
│   └── Plain Monte Carlo pricing for arithmetic Asian options
│
├── geometric_asian.py
│   └── Closed-form pricing for geometric Asian options
│
├── control_variate.py
│   └── Arithmetic Asian Monte Carlo with geometric Asian control variate
│
├── approximation.py
│   └── Turnbull-Wakeman and Levy analytical approximations
│
├── compare_methods.py
│   └── Comparison across Monte Carlo, control variate, and approximations
│
├── build_tw_residual_dataset.py
│   └── Generate residual dataset between TW approximation and CV benchmark
│
├── robustness_check.py
│   └── Robustness tests for the residual correction model
│
├── tw_residual_dataset_s0_grid.csv
│   └── Dataset of option parameters, benchmark prices, TW prices, and residuals
│
├── tw_bias_correction_random_test_results.csv
│   └── Prediction-level results for the random test set
│
└── bias_correction_summary.csv
    └── Summary of robustness test performance
```

---

### 3. Models Implemented

### 3.1 Black-Scholes GBM Simulation

The underlying asset follows the risk-neutral geometric Brownian motion model:

$$dS_t = rS_t\,dt + \sigma S_t\,dW_t.$$

Equivalently:

$$\frac{dS_t}{S_t} = r\,dt + \sigma\,dW_t.$$

The exact solution is:

$$S_t = S_0\exp\left[\left(r-\frac{1}{2}\sigma^2\right)t+\sigma W_t\right].$$

For a fixing date $t_i$, this gives:

$$S_{t_i} = S_0\exp\left[\left(r-\frac{1}{2}\sigma^2\right)t_i+\sigma W_{t_i}\right].$$

For discrete path simulation from $t_{i-1}$ to $t_i$, the exact transition update can be written as:

$$S_{t_i} = S_{t_{i-1}}\exp\left[\left(r-\frac{1}{2}\sigma^2\right)\Delta t + \sigma \sqrt{\Delta t}\,Z_i\right], \qquad Z_i \sim N(0,1).$$

> **Note:** The simulation is vectorized using NumPy for computational efficiency.

### 3.2 Plain Monte Carlo

The arithmetic Asian option price is estimated by:

$$\text{Price} = e^{-rT}\mathbb{E}\left[\max(A-K,0)\right].$$

Equivalently:

$$\text{Price} = e^{-rT}\mathbb{E}\left[(A-K)^+\right].$$

The plain Monte Carlo estimator is:

$$\widehat{V}_{\mathrm{arith}}^{\mathrm{MC}} = e^{-rT}\frac{1}{M}\sum_{j=1}^{M}\left(A^{(j)}-K\right)^+,$$

where $M$ is the number of simulated paths and $A^{(j)}$ is the arithmetic average along path $j$. 

### 3.3 Geometric Asian Closed-Form Formula

The geometric average is:

$$G = \left(\prod_{i=1}^{n} S_{t_i}\right)^{1/n}.$$

Under GBM, the logarithm of the geometric average is normally distributed, so the geometric Asian option has a closed-form solution. The geometric Asian option is used both as a standalone pricing benchmark and as a control variate for arithmetic Asian Monte Carlo pricing.

### 3.4 Control Variate Monte Carlo

The control variate estimator is defined as:

$$V_{\mathrm{CV}} = V_{\mathrm{arith}}^{\mathrm{MC}} - \beta\left(V_{\mathrm{geo}}^{\mathrm{MC}} - V_{\mathrm{geo}}^{\mathrm{closed}}\right).$$

where:
* $V_{\mathrm{arith}}^{\mathrm{MC}}$ is the simulated arithmetic Asian payoff.
* $V_{\mathrm{geo}}^{\mathrm{MC}}$ is the simulated geometric Asian payoff.
* $V_{\mathrm{geo}}^{\mathrm{closed}}$ is the analytical geometric Asian option price.
* $\beta$ is the control variate coefficient.

The optimal coefficient is:

$$\beta^* = \frac{\mathrm{Cov}\left(V_{\mathrm{arith}}^{\mathrm{MC}}, V_{\mathrm{geo}}^{\mathrm{MC}}\right)}{\mathrm{Var}\left(V_{\mathrm{geo}}^{\mathrm{MC}}\right)}.$$

#### Example: Variance Reduction

For the parameter setting $S_0 = 100$, $K = 100$, $r = 0.05$, $\sigma = 0.2$, $T = 1.0$, $n = 12$:

| Option Type | Plain MC Price | CV Price | Plain MC Std. Error | CV Std. Error | Geometric Asian Closed-Form | Beta |
|---|---:|---:|---:|---:|---:|---:|
| **Call** | 6.179168 | 6.156463 | 0.012055 | 0.000338 | 5.940200 | 1.0316 |
| **Put** | 3.523196 | 3.534525 | 0.007849 | 0.000199 | 3.651734 | 0.9761 |

The results show that using the geometric Asian option as a control variate reduces the Monte Carlo variance by more than **1000x** for both calls and puts in this example. This CV estimator is used as the benchmark price in the residual correction experiments.

### 3.5 Turnbull-Wakeman Approximation

#### 1. Core Concept
This method uses moment matching to approximate the arithmetic average $A$ as a lognormal random variable $\widetilde{A}$:

$$A \approx \widetilde{A}, \qquad \widetilde{A} \sim \mathrm{Lognormal}(\mu_A,\sigma_A^2).$$

The parameters $\mu_A$ and $\sigma_A^2$ match the first two moments of the true arithmetic average:

$$\mathbb{E}[\widetilde{A}] = \mathbb{E}[A], \qquad \mathrm{Var}(\widetilde{A}) = \mathrm{Var}(A).$$

#### 2. Parameter Calculation
Under the lognormal assumption, the parameter $\sigma_A^2$ is calculated as:

$$\sigma_A^2 = \ln\left(1+ \frac{\mathrm{Var}(A)}{\mathbb{E}[A]^2}\right).$$

> **Note:** The $\sigma_A^2$ here already incorporates the time dimension, representing total log-variance. When calculating $d_1$, the denominator is $\sigma_A$, not $\sigma_A\sqrt{T}$.

#### 3. Pricing Formula
The resulting call option price takes a Black-Scholes-like analytical form:

$$C_{\mathrm{TW}} = e^{-rT}\left(\mathbb{E}[A]\Phi(d_1) - K\Phi(d_2)\right).$$

where:

$$d_1 = \frac{\ln\left(\frac{\mathbb{E}[A]}{K}\right) + \frac{1}{2}\sigma_A^2}{\sigma_A}, \qquad d_2 = d_1-\sigma_A.$$

#### Example: Pricing Results Across Methods

For the parameter setting $S_0 = 100$, $K = 100$, $r = 0.05$, $\sigma = 0.2$, $T = 1.0$, $n = 12$:

| Option Type | Plain MC | Control Variate MC | Turnbull-Wakeman | Levy Approximation | Geometric Asian Closed-Form |
|---|---:|---:|---:|---:|---:|
| **Call** | 6.179168 | 6.156463 | 6.174171 | 5.782838 | 5.940200 |
| **Put** | 3.523196 | 3.534525 | 3.552611 | 3.364630 | 3.651734 |

---

## 4. Bias Correction Idea

Although the Turnbull-Wakeman approximation is computationally efficient, it exhibits systematic biases depending on the parameter space. This project uses a penalized regression model to learn and compensate for these errors.

The residual is defined as the TW approximation minus the CV benchmark:

$$\varepsilon_{\mathrm{TW}} = C_{\mathrm{TW}} - V_{\mathrm{CV}}.$$

* **Positive residual:** TW overprices the option relative to the benchmark.
* **Negative residual:** TW underprices the option.

The corrected price is then:

$$C_{\mathrm{corrected}} = C_{\mathrm{TW}} - \widehat{\varepsilon}_{\mathrm{TW}}.$$

This approach offers two primary advantages:
1. The analytical approximation already captures most of the option pricing structure.
2. The regression model only needs to learn the remaining systematic approximation error, which is simpler than learning the entire pricing function from scratch.

### 5. Polynomial Ridge Regression Residual Model

The model learns a mapping from option features to the scaled Turnbull-Wakeman residual:

$$\widehat{y} = f_\theta(x).$$

The base feature vector is:

$$x = \left[ \ln\left(\frac{K}{S_0}\right), \ \sigma, \ T, \ \frac{1}{n}, \ \frac{1}{\sqrt{n}} \right]^\top.$$

To capture nonlinear interactions, the base features are expanded into polynomial features:

$$\phi_d(x) = \left[ 1,\ x_1,\ldots,x_p,\ x_1^2,\ x_1x_2,\ldots,\ x_p^d \right]^\top.$$

The Ridge regression model is then fitted in the polynomial feature space:

$$\widehat{y} = \beta_0 + \phi_d(x)^\top \beta.$$

The coefficients are estimated by minimizing the penalized least-squares objective:

$$\min_{\beta_0,\beta} \sum_{i=1}^{N} \left( y_i - \beta_0 - \phi_d(x_i)^\top \beta \right)^2 + \lambda \lVert \beta \rVert_2^2.$$

After predicting the scaled residual $\widehat{y}$, the unscaled residual estimate is $\widehat{\varepsilon}_{\mathrm{TW}} = S_0 \widehat{y}$. The final corrected price is:

$$C_{\mathrm{corrected}} = C_{\mathrm{TW}} - S_0\widehat{y}.$$

---

## 6. Dataset

The current dataset contains **3360** parameter combinations. The grid includes multiple spot levels:

$$S_0 \in \{80,90,100,110,120\}.$$

For each spot level, the strike is generated through a fixed moneyness grid $m = \frac{K}{S_0}$:

$$m \in \{0.7,0.8,0.9,1.0,1.1,1.2,1.3\}.$$

The dataset also varies maturity, volatility, and monitoring frequency:

* $T \in \{0.25,0.5,1.0,2.0\}$
* $\sigma \in \{0.1,0.2,0.3,0.4,0.5,0.6\}$
* $n \in \{12,26,52,126\}$

**Total combinations:** 5 × 4 × 7 × 6 × 4 = 3360.

For each parameter combination, the dataset tracks benchmark prices, TW prices, residuals, scaled residuals, and engineered features.

---

## 7. Main Results

### 7.1 Random Test Set

The residual correction model performs strongly on a random train-test split.

| Metric | Original TW | Corrected TW | Reduction |
|---|---:|---:|---:|
| **MAE** | 0.069124 | 0.014550 | 78.95% |
| **RMSE** | 0.160030 | 0.019610 | 87.75% |
| **Max absolute error** | 0.861101 | 0.066608 | 92.26% |

**Additional Results:**
* **R² (test):** 0.983089
* **Improved fraction:** 59.29%

### 7.2 K-Fold Cross Validation

Five-fold cross validation on the scaled residual gives stable performance:

* **Fold MAE scores:** `[0.00014658, 0.00013989, 0.00016212, 0.00015171, 0.00014890]`
* **CV MAE mean:** 0.00014984
* **CV MAE std:** 0.00000728

The low standard deviation across folds suggests that the residual surface is learnable and stable.

### 7.3 Checkerboard Interpolation Test

A stricter checkerboard holdout was used across strike, volatility, and monitoring frequency.

| Metric | Original TW | Corrected TW | Reduction |
|---|---:|---:|---:|
| **MAE** | 0.080729 | 0.014847 | 81.61% |
| **RMSE** | 0.177457 | 0.019907 | 88.78% |
| **Max absolute error** | 0.870979 | 0.070214 | 91.94% |

**Additional Results:**
* **R² (test):** 0.985489
* **Improved fraction:** 61.07%

This shows that the Turnbull-Wakeman residual has a smooth structure that can be learned well under interpolation within the parameter grid.

---

## 8. Robustness Tests

Several holdout tests were performed to evaluate generalization across specific regimes.

### 8.1 High Volatility Holdout ($\sigma = 0.6$)
* **MAE Reduction:** 78.43% (0.257675 $\rightarrow$ 0.055593)
* **RMSE Reduction:** 77.57%
* **R² (test):** 0.915416 | **Improved fraction:** 88.39%

### 8.2 High Monitoring Frequency Holdout ($n = 126$)
* **MAE Reduction:** 81.04% (0.083010 $\rightarrow$ 0.015735)
* **RMSE Reduction:** 88.35%
* **R² (test):** 0.984449 | **Improved fraction:** 60.00%

### 8.3 ATM Holdout ($K = 100$)
* **MAE Reduction:** 85.92% (0.090687 $\rightarrow$ 0.012768)
* **RMSE Reduction:** 91.54%
* **R² (test):** 0.990697 | **Improved fraction:** 63.75%

### 8.4 Long Maturity Holdout ($T = 2.0$)
* **MAE Reduction:** 61.15% (0.198503 $\rightarrow$ 0.077116)
* **RMSE Reduction:** 75.28%
* **R² (test):** 0.909463 | **Improved fraction:** 45.71%

### 8.5 Leave-One-Moneyness Tests
The strongest results occur in the interior moneyness region ($m \in [0.9, 1.1]$), while performance drops at boundary moneyness levels ($m=0.7$ and $m=1.3$) where the model must extrapolate.

* **For $m = 0.9$:** MAE reduction = 88.07%, RMSE reduction = 92.17%, R² = 0.991392
* **For $m = 1.0$:** MAE reduction = 85.92%, RMSE reduction = 91.54%, R² = 0.990697
* **For $m = 1.1$:** MAE reduction = 75.58%, RMSE reduction = 87.28%, R² = 0.982341
* **For $m = 0.7$:** MAE reduction = 10.76%, Improved fraction = 30.00%
* **For $m = 1.3$:** MAE reduction = 9.69%, Improved fraction = 38.75%

---

## 9. Interpretation

The Turnbull-Wakeman approximation error is not random; it has a structured pattern. The residual correction model is highly effective when interpolating within the calibrated parameter grid, substantially reducing MAE, RMSE, and maximum absolute error. It struggles primarily with extreme boundary extrapolation (like long maturities or deep out-of-the-money bounds). Therefore, the current model should be viewed as a strong interpolation-based correction method.

## 10. Conservative Shrinkage Correction

A possible improvement is to apply a conservative shrinkage correction to prevent overcorrection in extrapolated regimes:

$$C_{\mathrm{shrink}} = C_{\mathrm{TW}} - \lambda \widehat{\varepsilon}_{\mathrm{TW}}, \qquad 0 \leq \lambda \leq 1.$$

The original full correction corresponds to $\lambda = 1$. A smaller value (e.g., $\lambda = 0.5$ or $0.75$) can be tuned on a validation set and used as a robustness diagnostic.

## 11. Current Conclusion

This project shows that a data-driven residual correction can significantly improve the Turnbull-Wakeman approximation. The main contribution is to combine:

**Financial Approximation + Control Variate Benchmark + Residual Learning**

The corrected approximation keeps the speed advantage of analytical methods while moving closer to the accuracy of control variate Monte Carlo. The strongest evidence comes from the checkerboard interpolation test, reducing MAE by approximately **82%** with an out-of-sample R² of **0.985**.

## 12. Future Work

1.  Expand the training grid to include more extreme moneyness levels ($m \in \{0.6, \ldots, 1.4\}$).
2.  Add more spot levels for additional scale robustness tests ($S_0 \in \{70, \ldots, 130\}$).
3.  Compare full residual correction with conservative shrinkage correction.
4.  Add residual heatmaps over moneyness and volatility.
5.  Extend the framework to Greeks (Delta, Gamma, Vega, Rho, Theta).
6.  Compare different residual models (spline interpolation, Gaussian process, gradient boosting).
7.  Test under stochastic volatility or local volatility assumptions.

---

## 13. Key Takeaway

* **Turnbull-Wakeman approximation:** Fast but biased.
* **Control variate Monte Carlo:** Accurate but computationally expensive.
* **Bias-Corrected TW:** Learns the structured residual, achieving high accuracy for interpolation while preserving analytical speeds.

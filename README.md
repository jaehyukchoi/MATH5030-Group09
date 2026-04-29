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

## 3. Models Implemented

### 3.1 Black-Scholes GBM Simulation

The underlying asset follows the risk-neutral geometric Brownian motion model:

$$
dS_t = rS_t\,dt + \sigma S_t\,dW_t.
$$

Equivalently,

$$
\frac{dS_t}{S_t} = r\,dt + \sigma\,dW_t.
$$

The exact solution is

$$
S_t =
S_0
\exp\left[
\left(r-\frac{1}{2}\sigma^2\right)t+\sigma W_t
\right].
$$

For a fixing date $t_i$, this gives

$$
S_{t_i}
=
S_0
\exp\left[
\left(r-\frac{1}{2}\sigma^2\right)t_i+\sigma W_{t_i}
\right].
$$


---

### 3.2 Plain Monte Carlo

The arithmetic Asian option price is estimated by

$$
\text{Price}
=
e^{-rT}
\mathbb{E}
\left[
\max(A-K,0)
\right].
$$

Equivalently,

$$
\text{Price}
=
e^{-rT}
\mathbb{E}
\left[
(A-K)^+
\right].
$$

The plain Monte Carlo estimator is

$$
\widehat{V}_{\mathrm{arith}}^{\mathrm{MC}}
=
e^{-rT}
\frac{1}{M}
\sum_{j=1}^{M}
\left(A^{(j)}-K\right)^+,
$$

where $M$ is the number of simulated paths and $A^{(j)}$ is the arithmetic average along path $j$.

Plain Monte Carlo is simple but can have relatively high variance, especially when many benchmark prices are required across a parameter grid.

---

### 3.3 Geometric Asian Closed-Form Formula

The geometric average is

$$
G
=
\left(
\prod_{i=1}^{n} S_{t_i}
\right)^{1/n}.
$$

Under GBM, the logarithm of the geometric average is normally distributed, so the geometric Asian option has a closed-form solution.

The geometric Asian option is used both as:

1. A standalone pricing benchmark for geometric Asian options.
2. A control variate for arithmetic Asian Monte Carlo pricing.

---

### 3.4 Control Variate Monte Carlo

The control variate estimator is defined as

$$
V_{\mathrm{CV}}
=
V_{\mathrm{arith}}^{\mathrm{MC}}
-
\beta
\left(
V_{\mathrm{geo}}^{\mathrm{MC}}
-
V_{\mathrm{geo}}^{\mathrm{closed}}
\right).
$$

where:

- $V_{\mathrm{arith}}^{\mathrm{MC}}$ is the simulated arithmetic Asian payoff.
- $V_{\mathrm{geo}}^{\mathrm{MC}}$ is the simulated geometric Asian payoff.
- $V_{\mathrm{geo}}^{\mathrm{closed}}$ is the analytical geometric Asian option price.
- $\beta$ is the control variate coefficient.

The optimal coefficient is

$$
\beta^*
=
\frac{
\mathrm{Cov}
\left(
V_{\mathrm{arith}}^{\mathrm{MC}},
V_{\mathrm{geo}}^{\mathrm{MC}}
\right)
}{
\mathrm{Var}
\left(
V_{\mathrm{geo}}^{\mathrm{MC}}
\right)
}.
$$

Because arithmetic and geometric Asian payoffs are highly correlated, this greatly reduces Monte Carlo variance.

#### Example: Variance Reduction

For the parameter setting

$$
S_0 = 100,\quad K = 100,\quad r = 0.05,\quad \sigma = 0.2,\quad T = 1.0,\quad n = 12,
$$

the control variate estimator significantly reduces the Monte Carlo standard error.

| Option Type | Plain MC Price | CV Price | Plain MC Std. Error | CV Std. Error | Geometric Asian Closed-Form | Beta |
|---|---:|---:|---:|---:|---:|---:|
| Call | $6.179168$ | $6.156463$ | $0.012055$ | $0.000338$ | $5.940200$ | $1.0316$ |
| Put | $3.523196$ | $3.534525$ | $0.007849$ | $0.000199$ | $3.651734$ | $0.9761$ |

The results show that using the geometric Asian option as a control variate reduces the Monte Carlo variance by more than $1000\times$ for both calls and puts in this example.

The control variate estimator is used as the benchmark price in the residual correction experiments.

---

### 3.5 Turnbull-Wakeman Approximation

The Turnbull-Wakeman approximation is one of the most classic analytical approximation methods used for pricing Asian options.

#### 1. Core Concept

This method uses moment matching to approximate the arithmetic average $A$ as a lognormal random variable $\widetilde{A}$:

$$
A \approx \widetilde{A},
\qquad
\widetilde{A} \sim \mathrm{Lognormal}(\mu_A,\sigma_A^2).
$$

The parameters $\mu_A$ and $\sigma_A^2$ are chosen to match the first two moments of the true arithmetic average:

$$
\mathbb{E}[\widetilde{A}]
=
\mathbb{E}[A],
\qquad
\mathrm{Var}(\widetilde{A})
=
\mathrm{Var}(A).
$$

#### 2. Parameter Calculation

Under the lognormal assumption, the parameter $\sigma_A^2$, which is the variance of $\ln \widetilde{A}$ used in the pricing formula, is calculated as

$$
\sigma_A^2
=
\ln
\left(
1+
\frac{
\mathrm{Var}(A)
}{
\mathbb{E}[A]^2
}
\right).
$$

The $\sigma_A^2$ here already incorporates the time dimension, so it represents total log-variance. When calculating $d_1$, the denominator is $\sigma_A$, not $\sigma_A\sqrt{T}$.

#### 3. Pricing Formula

The resulting call option price takes a Black-Scholes-like analytical form:

$$C_{\mathrm{TW}} = e^{-rT} \left( \mathbb{E}[A]\Phi(d_1) - K\Phi(d_2) \right).$$

where

$$d_1 = \frac{ \ln\left(\frac{\mathbb{E}[A]}{K}\right) + \frac{1}{2}\sigma_A^2 }{ \sigma_A }, \qquad d_2 = d_1-\sigma_A.$$

Here, $\Phi(\cdot)$ is the standard normal cumulative distribution function.

#### Example: Pricing Results Across Methods

For the parameter setting

$$
S_0 = 100,\quad K = 100,\quad r = 0.05,\quad \sigma = 0.2,\quad T = 1.0,\quad n = 12,
$$

the pricing results across different methods are:

| Option Type | Plain MC | Control Variate MC | Turnbull-Wakeman | Levy Approximation | Geometric Asian Closed-Form |
|---|---:|---:|---:|---:|---:|
| Call | $6.179168$ | $6.156463$ | $6.174171$ | $5.782838$ | $5.940200$ |
| Put | $3.523196$ | $3.534525$ | $3.552611$ | $3.364630$ | $3.651734$ |

---
## Figures

### TW Approximation Error vs Strike

![TW approximation error vs strike](figures/error_vs_strike_sigma_0p2_n12.png)

### TW Approximation Error vs Volatility

![TW approximation error vs volatility](figures/error_vs_vol_K100_n12.png)

### Runtime vs Monitoring Frequency

![Runtime vs monitoring frequency](figures/runtime_vs_monitoring_frequency_K100_sigma_0p2.png)


## 4. Bias Correction Idea

Although the Turnbull-Wakeman approximation is computationally efficient, it exhibits systematic biases depending on the parameter space. This project uses a penalized regression model to learn and compensate for these systematic errors.

In this project, the residual is defined as the Turnbull-Wakeman approximation minus the control variate benchmark:

$$\varepsilon_{\mathrm{TW}} = C_{\mathrm{TW}} - V_{\mathrm{CV}}.$$

A positive residual means that the Turnbull-Wakeman approximation overprices the option relative to the control variate benchmark. A negative residual means that it underprices the option.

The corrected price is then

$$C_{\mathrm{corrected}} = C_{\mathrm{TW}} - \widehat{\varepsilon}_{\mathrm{TW}}.$$
This approach offers two primary advantages:

1. The analytical approximation already captures most of the option pricing structure.
2. The regression model only needs to learn the remaining systematic approximation error, which is simpler than learning the entire pricing function from scratch.

In essence, the model does not function as a black-box pricer. Instead, it acts as a residual correction layer integrated into a traditional financial approximation framework.

---

### Polynomial Ridge Regression Residual Model

The model learns a mapping from option features to the scaled Turnbull-Wakeman residual:

$$
\widehat{y}
=
f_\theta(x).
$$

The base feature vector is

$$
x
=
\left[
\ln\left(\frac{K}{S_0}\right),
\ \sigma,
\ T,
\ \frac{1}{n}
\
\right]^\top.
$$

To capture nonlinear interactions in the residual surface, the base features are expanded into polynomial features:

$$
\phi_d(x)
=
\left[
1,\ x_1,\ldots,x_p,\ x_1^2,\ x_1x_2,\ldots,\ x_p^d
\right]^\top,
$$

where \(d\) is the polynomial degree and \(p\) is the number of base features.

The Ridge regression model is then fitted in the polynomial feature space:

$$
\widehat{y}
=
\beta_0 + \phi_d(x)^\top \beta.
$$

The coefficients are estimated by minimizing the penalized least-squares objective:

$$
\min_{\beta_0,\beta} \sum_{i=1}^{N} \left( y_i - \beta_0 - \phi_d(x_i)^\top \beta \right)^2 + \lambda \lVert \beta \rVert_2^2.
$$

The intercepts not penalized. The Ridge penalty shrinks the polynomial coefficients and helps reduce overfitting, especially when higher-order interaction terms are included.

After predicting the scaled residual , the unscaled residual estimate is

$$
\widehat{\varepsilon}_{\mathrm{TW}}
=
S_0 \widehat{y}.
$$

Since the residual is defined as

$$\varepsilon_{\mathrm{TW}} = C_{\mathrm{TW}} - V_{\mathrm{CV}},$$

the corrected Turnbull-Wakeman price is

$$C_{\mathrm{corrected}} = C_{\mathrm{TW}} - \widehat{\varepsilon}_{\mathrm{TW}}.$$

Equivalently,

$$C_{\mathrm{corrected}} = C_{\mathrm{TW}} - S_0\widehat{y}.$$
### Dataset

The current dataset contains $3360$ parameter combinations.

The grid includes multiple spot levels:

$$
S_0 \in \{80,90,100,110,120\}.
$$

For each spot level, the strike is generated through a fixed moneyness grid:

$$
m
=
\frac{K}{S_0},
\qquad
m
\in
\{0.7,0.8,0.9,1.0,1.1,1.2,1.3\}.
$$

Equivalently,

$$
K = S_0m.
$$

The dataset also varies maturity, volatility, and monitoring frequency:

$$
T \in \{0.25,0.5,1.0,2.0\},
$$

$$
\sigma \in \{0.1,0.2,0.3,0.4,0.5,0.6\},
$$

$$
n \in \{12,26,52,126\}.
$$

Therefore, the total number of parameter combinations is

$$
5 \times 4 \times 7 \times 6 \times 4 = 3360.
$$

For each parameter combination, the following quantities are computed:

- Control variate benchmark price $V_{\mathrm{CV}}$
- Turnbull-Wakeman approximation price $C_{\mathrm{TW}}$
- Turnbull-Wakeman residual $\varepsilon_{\mathrm{TW}} = C_{\mathrm{TW}} - V_{\mathrm{CV}}$
- Scaled residual $y = \varepsilon_{\mathrm{TW}} / S_0$
- Option parameters and engineered features

Because the dataset includes multiple spot levels, it can now be used to test scale robustness through leave-one-$S_0$ experiments.

---

## Robustness Check Results

This section evaluates the robustness of the proposed Turnbull-Wakeman bias-correction framework.

The model learns the scale-normalized residual

$$
y =
\frac{
C_{\mathrm{TW}} - C_{\mathrm{CV}}
}{
S_0
},
$$

where $C_{\mathrm{TW}}$ is the Turnbull-Wakeman approximation and $C_{\mathrm{CV}}$ is the control-variate Monte Carlo benchmark.

The correction is then applied as

$$C_{\mathrm{corrected}} = C_{\mathrm{TW}} - S_0 \widehat{y}.$$

The model uses the following economically motivated features:

$$
\log(K/S_0), \qquad \sigma, \qquad T, \qquad \frac{1}{n}.
$$

A third-degree polynomial Ridge regression is used to capture nonlinear interactions while controlling overfitting.

---

### Dataset

The robustness dataset is built on the following parameter grid:

| Parameter | Values |
|---|---|
| Spot \(S_0\) | 80, 90, 100, 110, 120 |
| Moneyness \(K/S_0\) | 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3 |
| Volatility \(\sigma\) | 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 |
| Maturity \(T\) | 0.25, 0.5, 1.0, 2.0 |
| Monitoring dates \(n\) | 12, 26, 52, 126 |

Total number of cases:

$$
5 \times 7 \times 6 \times 4 \times 4 = 3360.
$$

For each parameter combination, the control-variate Monte Carlo price is used as the benchmark.

---

## Main Robustness Summary

| Test | Test Size | Test \(R^2\) | Original TW MAE | Corrected TW MAE | MAE Reduction | Improved Cases |
|---|---:|---:|---:|---:|---:|---:|
| Random Train/Test Split | 840 | 0.9912 | 0.091455 | 0.012304 | **86.55%** | 66.90% |
| High Volatility Holdout, \(\sigma=0.6\) | 560 | 0.9856 | 0.256846 | 0.022913 | **91.08%** | 93.04% |
| Long Maturity Holdout, \(T=2.0\) | 840 | 0.9625 | 0.228149 | 0.042074 | **81.56%** | 68.10% |
| High-\(n\) Holdout, \(n=126\) | 840 | 0.9901 | 0.095214 | 0.013296 | **86.04%** | 70.48% |
| Checkerboard Interpolation | 1680 | 0.9905 | 0.093009 | 0.012496 | **86.56%** | 68.39% |

The correction framework performs strongly across the main robustness tests. In most cases, the corrected Turnbull-Wakeman approximation reduces MAE by more than 80%.

---

## K-Fold Cross Validation

The five-fold cross validation is performed on the scaled residual target.

| Fold | Scaled MAE |
|---:|---:|
| 1 | 0.00012021 |
| 2 | 0.00012914 |
| 3 | 0.00012479 |
| 4 | 0.00011789 |
| 5 | 0.00012160 |
| **Mean** | **0.00012273** |
| **Std** | **0.00000391** |

The small standard deviation indicates that the random train-test result is not due to a lucky split. The model is stable across different random partitions.

---

## Leave-One-\(S_0\)-Out Robustness

In this test, one spot level is completely removed from the training set and used only for testing.

| Held-out \(S_0\) | Test Size | Test \(R^2\) | Original TW MAE | Corrected TW MAE | MAE Reduction | Improved Cases |
|---:|---:|---:|---:|---:|---:|---:|
| 80 | 672 | 0.9911 | 0.074612 | 0.009795 | **86.87%** | 69.05% |
| 90 | 672 | 0.9915 | 0.083539 | 0.010692 | **87.20%** | 69.35% |
| 100 | 672 | 0.9911 | 0.092727 | 0.012126 | **86.92%** | 69.20% |
| 110 | 672 | 0.9913 | 0.102198 | 0.013155 | **87.13%** | 69.20% |
| 120 | 672 | 0.9910 | 0.111560 | 0.014665 | **86.86%** | 69.79% |

The leave-one-\(S_0\)-out results are very stable. The MAE reduction remains around 87% for every held-out spot level. This supports the scale-normalized residual design:

$$
\frac{C_{\mathrm{TW}} - C_{\mathrm{CV}}}{S_0}.
$$

---

## Leave-One-Moneyness-Out Robustness

In this test, one moneyness level is completely removed from the training set and used only for testing.

| Held-out Moneyness \(K/S_0\) | Test Size | Test \(R^2\) | Original TW MAE | Corrected TW MAE | MAE Reduction | Improved Cases |
|---:|---:|---:|---:|---:|---:|---:|
| 0.7 | 480 | 0.7773 | 0.104288 | 0.082079 | **21.30%** | 33.33% |
| 0.8 | 480 | 0.9839 | 0.133809 | 0.023001 | **82.81%** | 58.33% |
| 0.9 | 480 | 0.9906 | 0.138480 | 0.015300 | **88.95%** | 83.75% |
| 1.0 | 480 | 0.9928 | 0.103753 | 0.011799 | **88.63%** | 73.96% |
| 1.1 | 480 | 0.9858 | 0.066591 | 0.014046 | **78.91%** | 69.58% |
| 1.2 | 480 | 0.9709 | 0.055000 | 0.014226 | **74.14%** | 73.54% |
| 1.3 | 480 | 0.6796 | 0.048568 | 0.038396 | **20.94%** | 44.58% |

The model performs well for interior moneyness levels but is weaker at the boundary levels \(0.7\) and \(1.3\).

This is expected because holding out \(0.7\) or \(1.3\) is a boundary extrapolation test:

- Holding out \(0.7\): the model only sees \(K/S_0 >= 0.8\) during training.
- Holding out \(1.3\): the model only sees \(K/S_0 <=1.2\) during training.

Therefore, the framework interpolates well inside the moneyness grid but is less reliable when extrapolating beyond the observed moneyness range.

---

## Moneyness Interpolation vs Boundary Extrapolation

| Region | Held-out Moneyness | Average MAE Reduction | Average Test \(R^2\) | Interpretation |
|---|---|---:|---:|---|
| Boundary Extrapolation | 0.7, 1.3 | **21.12%** | 0.7285 | Weak extrapolation at grid edges |
| Interior Moneyness | 0.8, 0.9, 1.0, 1.1, 1.2 | **82.69%** | 0.9848 | Strong interpolation inside the grid |

This is the main limitation of the current framework. The correction surface is smooth and learnable inside the training grid, but boundary extrapolation remains challenging.

---
## Robustness Figures

### Original vs Corrected TW Error

![Original vs corrected TW error](figures/random_original_vs_corrected_tw_error.png)

### Mean Absolute Error Reduction

![TW vs corrected TW MAE](figures/random_tw_vs_corrected_mae.png)

### Robustness MAE Reduction Summary

![Robustness MAE reduction summary](figures/robustness_mae_reduction_summary.png)
## Key Findings

| Finding | Evidence |
|---|---|
| The correction framework is effective overall | Random split reduces MAE by **86.55%** |
| Results are stable under random partitions | K-fold CV mean scaled MAE = **0.00012273**, std = **0.00000391** |
| High-volatility regime is handled well | \(\sigma=0.6\) holdout reduces MAE by **91.08%** |
| Long maturity is harder but still improved | \(T=2.0\) holdout reduces MAE by **81.56%** |
| Scaling by \(S_0\) works well | Leave-one-\(S_0\)-out reductions are all around **87%** |
| The model interpolates smoothly across the grid | Checkerboard reduction = **86.56%** |
| Main weakness is boundary moneyness extrapolation | \(K/S_0=0.7\) and \(1.3\) reduce MAE by only about **21%** |

---

## Cubic interpolation method

The cubic interpolation result on the original residual grid is an in-sample reconstruction test, because the interpolator is evaluated on the same grid used to build it. To obtain a more objective evaluation, we construct a separate off-grid test set.

The off-grid test points lie inside the calibrated parameter domain but are not part of the original residual grid. For example, if the original grid contains moneyness values such as

$$0.7, 0.8, 0.9, \ldots, 1.3,$$

the off-grid test uses intermediate values such as

$$0.75, 0.85, 0.95, \ldots, 1.25.$$

For each off-grid test case, a fresh control-variate Monte Carlo benchmark is computed. We then compare three methods:

1. Original Turnbull-Wakeman approximation.
2. Polynomial Ridge residual correction.
3. Cubic interpolation residual correction.

The residual correction is based on the scaled residual

$$\frac{C_{\mathrm{TW}} - C_{\mathrm{CV}}}{S_0},$$

and the corrected price is computed as

$$C_{\mathrm{corrected}}=C_{\mathrm{TW}}-S_0 \widehat{f}(m,\sigma,T,n),$$

where

$$m=\frac{K}{S_0}.$$

### Off-Grid Test Results

| Method | MAE | RMSE | Max Abs Error | MAE Reduction | RMSE Reduction | Max Error Reduction | Improved Fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original TW | 0.074885 | 0.128262 | 0.554559 | — | — | — | — |
| Polynomial Ridge Corrected TW | 0.011538 | 0.013757 | **0.035299** | 84.59% | 89.27% | **93.63%** | 73.33% |
| Cubic Interpolation Corrected TW | **0.003505** | **0.006475** | 0.070887 | **95.32%** | **94.95%** | 87.22% | **88.64%** |

### Interpretation

The off-grid test shows that cubic residual interpolation is highly effective for pricing new parameter points inside the calibrated grid. It achieves the lowest MAE and RMSE, reducing the original Turnbull-Wakeman MAE by **95.32%** and RMSE by **94.95%**.

Compared with polynomial Ridge regression, cubic interpolation has stronger average accuracy:

* Polynomial Ridge MAE: `0.011538`
* Cubic interpolation MAE: `0.003505`

The cubic interpolation method also improves a larger fraction of individual test cases:

* Polynomial Ridge improved fraction: `73.33%`
* Cubic interpolation improved fraction: `88.64%`

However, polynomial Ridge achieves a lower maximum absolute error:

* Polynomial Ridge max absolute error: `0.035299`
* Cubic interpolation max absolute error: `0.070887`

This suggests a useful trade-off:

| Method | Strength | Weakness |
|---|---|---|
| Polynomial Ridge correction | Better worst-case error control | Higher average error |
| Cubic interpolation correction | Better average accuracy inside the calibrated grid | Larger worst-case error due to possible interpolation overshoot |

Overall, the off-grid test suggests that the residual surface is smooth and can be accurately interpolated inside the calibrated parameter domain. Polynomial Ridge is more conservative and robust in worst-case error control, while cubic interpolation is more accurate for grid-interior pricing.

## Conclusion

The proposed Turnbull-Wakeman bias-correction framework is robust across most tested regimes. The correction reduces pricing MAE by approximately 81% to 91% in the main robustness tests, including random train-test split, high-volatility holdout, long-maturity holdout, high-monitoring-frequency holdout, leave-one-\(S_0\)-out, and checkerboard interpolation.

The strongest evidence comes from the leave-one-\(S_0\)-out tests, where the MAE reduction remains around 87% across all held-out spot levels. This confirms that modeling the scaled residual is effective for generalizing across spot levels.

The main limitation is boundary moneyness extrapolation. When the extreme moneyness levels \(0.7\) and \(1.3\) are entirely excluded from training, performance drops significantly. In contrast, interior moneyness levels from \(0.8\) to \(1.2\) show strong interpolation performance.

Overall, the results suggest that the residual-learning correction captures a stable and systematic structure in the Turnbull-Wakeman approximation error.

## 5. Future Work

Planned extensions include:

1. Expand the training grid to include more extreme moneyness levels:

$$
m \in \{0.6,0.7,\ldots,1.4\}.
$$

2. Add more spot levels for additional scale robustness tests:

$$
S_0 \in \{70,80,90,100,110,120,130\}.
$$

3. Compare full residual correction with conservative shrinkage correction.


4. Extend the framework to Greeks such as Delta, Gamma, Vega, Rho, and Theta.


5. Test the method under different market assumptions, such as stochastic volatility or local volatility.

---

## 6. Key Takeaway

The Turnbull-Wakeman approximation is fast but biased.

Control variate Monte Carlo is accurate but computationally expensive.

This project shows that the bias of the Turnbull-Wakeman approximation is learnable. By modeling the residual relative to a control variate benchmark, the corrected approximation achieves much better accuracy while preserving the speed of an analytical approximation.

The method is especially effective for interpolation within a calibrated parameter grid, while extrapolation to extreme moneyness remains the main challenge.
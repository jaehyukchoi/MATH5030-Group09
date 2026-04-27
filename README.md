# Bias-Corrected Turnbull-Wakeman Approximation for Arithmetic Asian Options

This project studies the pricing of arithmetic Asian options under the Black-Scholes geometric Brownian motion framework. Since arithmetic Asian options do not admit a simple closed-form solution, the project compares Monte Carlo methods, analytical approximations, and a data-driven residual correction approach.

The main goal is to build a pricing framework that keeps the speed of analytical approximations while improving their accuracy relative to a high-precision Monte Carlo benchmark.

---

## 1. Motivation

Asian options depend on the average price of the underlying asset over time. For a discretely monitored arithmetic Asian call option, the arithmetic average is

$$
A = \frac{1}{n}\sum_{i=1}^{n} S_{t_i}
$$

and the call payoff is

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
│   └── Generate residual dataset between CV benchmark and TW approximation
│
├── robustness_check.py
│   └── Robustness tests for the residual correction model
│
├── tw_residual_dataset.csv
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
S_t = S_0
\exp\left[
\left(r-\frac{1}{2}\sigma^2\right)t + \sigma W_t
\right].
$$

For a fixing date \(t_i\), this gives

$$
S_{t_i}
=
S_0
\exp\left[
\left(r-\frac{1}{2}\sigma^2\right)t_i
+
\sigma W_{t_i}
\right].
$$

In discrete simulation from \(t_{i-1}\) to \(t_i\), the update can be written as

$$
S_{t_i}
=
S_{t_{i-1}}
\exp\left[
\left(r-\frac{1}{2}\sigma^2\right)\Delta t
+
\sigma \sqrt{\Delta t}\,Z_i
\right],
\qquad
Z_i \sim N(0,1).
$$

The simulation is vectorized using NumPy for efficiency.

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

where \(M\) is the number of simulated paths and \(A^{(j)}\) is the arithmetic average along path \(j\).

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

The control variate estimator is

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
\right),
$$

where:

- \(V_{\mathrm{arith}}^{\mathrm{MC}}\) is the simulated arithmetic Asian payoff.
- \(V_{\mathrm{geo}}^{\mathrm{MC}}\) is the simulated geometric Asian payoff.
- \(V_{\mathrm{geo}}^{\mathrm{closed}}\) is the analytical geometric Asian price.
- \(\beta\) is the control variate coefficient.

The optimal coefficient is

$$
\beta^*
=
\frac{
\operatorname{Cov}
\left(
V_{\mathrm{arith}}^{\mathrm{MC}},
V_{\mathrm{geo}}^{\mathrm{MC}}
\right)
}{
\operatorname{Var}
\left(
V_{\mathrm{geo}}^{\mathrm{MC}}
\right)
}.
$$

Because arithmetic and geometric Asian payoffs are highly correlated, this greatly reduces Monte Carlo variance.

### Example: Variance Reduction

For the parameter setting

$$
S_0 = 100,\quad K = 100,\quad r = 0.05,\quad \sigma = 0.2,\quad T = 1.0,\quad n = 12,
$$

The control variate estimator significantly reduces the Monte Carlo standard error.

| Option Type | Plain MC Price | CV Price | Plain MC Std. Error | CV Std. Error | Geometric Asian Closed-Form | Beta |
|---|---:|---:|---:|---:|---:|---:|
| Call | \(6.179168\) | \(6.156463\) | \(0.012055\) | \(0.000338\) | \(5.940200\) | \(1.0316\) |
| Put | \(3.523196\) | \(3.534525\) | \(0.007849\) | \(0.000199\) | \(3.651734\) | \(0.9761\) |
The results show that using the geometric Asian option as a control variate reduces the Monte Carlo variance by more than \(1000\times\) for both calls and puts in this example.
The control variate estimator is used as the benchmark price in the residual correction experiments.


---

### 3.5 Turnbull-Wakeman Approximation

The Turnbull-Wakeman approximation is one of the most classic analytical approximation methods used for pricing Asian options.

#### 1. Core Concept
This method uses **moment matching** to approximate the arithmetic average $A$ as a lognormal random variable $\widetilde{A}$:

$$A \approx \widetilde{A}, \qquad \widetilde{A} \sim \operatorname{Lognormal}(\mu_A, \sigma_A^2)$$

The parameters $\mu_A$ and $\sigma_A^2$ are chosen to match the first two moments of the true arithmetic average:

$$\mathbb{E}[\widetilde{A}] = \mathbb{E}[A], \qquad \operatorname{Var}(\widetilde{A}) = \operatorname{Var}(A)$$

#### 2. Parameter Calculation (Crucial for Implementation)
Under the lognormal assumption, the parameter $\sigma_A^2$ (which is the variance of $\ln \widetilde{A}$) used in the pricing formula must be calculated via the following equation:

$$\sigma_A^2 = \ln\left( 1 + \frac{\operatorname{Var}(A)}{\mathbb{E}[A]^2} \right)$$

> **Note**: The $\sigma_A^2$ here already incorporates the time dimension (i.e., it represents the Total Variance). When calculating $d_1$ later, you do not need to multiply the denominator by $\sqrt{T}$.

#### 3. Pricing Formula
The resulting call option price takes a Black-Scholes-like analytical form:

$$C_{\mathrm{TW}} = e^{-rT} \left( \mathbb{E}[A]\Phi(d_1) - K\Phi(d_2) \right)$$

where:

$$d_1 = \frac{\ln\left( \frac{\mathbb{E}[A]}{K} \right) + \frac{1}{2}\sigma_A^2}{\sigma_A}, \qquad d_2 = d_1 - \sigma_A$$

### Example: Pricing Results Across Methods

For the parameter setting

$$
S_0 = 100,\quad K = 100,\quad r = 0.05,\quad \sigma = 0.2,\quad T = 1.0,\quad n = 12,
$$

the pricing results across different methods are:

| Option Type | Plain MC | Control Variate MC | Turnbull-Wakeman | Levy Approximation | Geometric Asian Closed-Form |
|---|---:|---:|---:|---:|---:|
| Call | \(6.179168\) | \(6.156463\) | \(6.174171\) | \(5.782838\) | \(5.940200\) |
| Put | \(3.523196\) | \(3.534525\) | \(3.552611\) | \(3.364630\) | \(3.651734\) |

## 4. Bias Correction Idea

Although the Turnbull-Wakeman (TW) approximation is computationally efficient, it exhibits systematic biases depending on the parameter space. We propose to establish a linear regression model with penalized term to learn and compensate for these systematic errors.

The residual (bias) is defined as:

$$\varepsilon_{\mathrm{TW}} = C_{\mathrm{TW}}-V_{\mathrm{true}}  $$

The corrected price is then:

$$C_{\mathrm{corrected}} = C_{\mathrm{TW}} - \widehat{\varepsilon}_{\mathrm{TW}}$$

This approach offers two primary advantages:

1. **Structural Integrity**: The analytical approximation already captures the majority of the option's pricing structure and Greeks.
2. **Reduced Complexity**: The regression model only needs to learn the remaining systematic approximation error, which is a much simpler mapping than learning the entire pricing function from scratch.

In essence, the model does not function as a "black-box" pricer. Instead, it acts as a **residual correction layer** integrated into a traditional financial approximation framework.

## 5. Feature Engineering

The residual correction model utilizes a compact set of financially meaningful features:

$$\text{Log-moneyness} = \ln\left(\frac{K}{S_0}\right), \quad \sigma, \quad T, \quad \frac{1}{n}, \quad \frac{1}{\sqrt{n}}$$

The target variable is the scaled residual:

$$y = \frac{V_{\mathrm{CV}} - C_{\mathrm{TW}}}{S_0}$$

Scaling by $S_0$ makes the residual dimensionless, establishing a robust framework that generalizes well across different underlying spot price levels.

## 6. Dataset

The current dataset contains \(3360\) parameter combinations.

The grid includes multiple spot levels:

$$
S_0 \in \{80,90,100,110,120\}.
$$

For each spot level, the strike is generated through a fixed moneyness grid:

$$
m = \frac{K}{S_0},
\qquad
m \in \{0.7,0.8,0.9,1.0,1.1,1.2,1.3\}.
$$

Equivalently,

$$
K = S_0 m.
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

* **Control variate benchmark price** $V_{\mathrm{CV}}$
* **Turnbull-Wakeman approximation price** $C_{\mathrm{TW}}$
* **Turnbull-Wakeman residual** $\varepsilon_{\mathrm{TW}} = C_{\mathrm{TW}} - V_{\mathrm{CV}}$
* **Scaled residual** $y = \frac{\varepsilon_{\mathrm{TW}}}{S_0}$
* **Option parameters and engineered features**
Because the dataset includes multiple spot levels, it can now be used to test scale robustness through leave-one-\(S_0\) experiments.

## 7. Main Results

### 7.1 Random Test Set

The residual correction model performs strongly on a random train-test split.

| Metric | Original TW | Corrected TW | Reduction |
|---|---:|---:|---:|
| MAE | \(0.069124\) | \(0.014550\) | \(78.95\%\) |
| RMSE | \(0.160030\) | \(0.019610\) | \(87.75\%\) |
| Max absolute error | \(0.861101\) | \(0.066608\) | \(92.26\%\) |

Additional results:

$$
R^2_{\mathrm{test}} = 0.983089,
\qquad
\text{Improved fraction} = 59.29\%.
$$

The correction significantly reduces average error and especially reduces large pricing errors.

The improved-case fraction is lower than the MAE and RMSE improvements because some cases where the original Turnbull-Wakeman approximation is already very accurate can be slightly overcorrected.

---

### 7.2 K-Fold Cross Validation

Five-fold cross validation on the scaled residual gives stable performance:

$$
\text{Fold MAE scores}
=
[0.00014658,\ 0.00013989,\ 0.00016212,\ 0.00015171,\ 0.00014890].
$$

The mean and standard deviation are

$$
\text{CV MAE mean} = 0.00014984,
\qquad
\text{CV MAE std} = 0.00000728.
$$

The low standard deviation across folds suggests that the residual surface is learnable and that the model performance is not driven by one lucky train-test split.

---

### 7.3 Checkerboard Interpolation Test

A stricter checkerboard holdout was used across strike, volatility, and monitoring frequency. This tests whether the model can infer missing grid points from nearby regions of the parameter space.

| Metric | Original TW | Corrected TW | Reduction |
|---|---:|---:|---:|
| MAE | \(0.080729\) | \(0.014847\) | \(81.61\%\) |
| RMSE | \(0.177457\) | \(0.019907\) | \(88.78\%\) |
| Max absolute error | \(0.870979\) | \(0.070214\) | \(91.94\%\) |

Additional results:

$$
R^2_{\mathrm{test}} = 0.985489,
\qquad
\text{Improved fraction} = 61.07\%.
$$

This is one of the most important results of the project. It shows that the Turnbull-Wakeman residual has a smooth structure that can be learned well under interpolation within the parameter grid.

---

## 8. Robustness Tests

Several holdout tests were performed to evaluate generalization across specific regimes.

### 8.1 High Volatility Holdout

Holding out \(\sigma=0.6\):

| Metric | Original TW | Corrected TW | Reduction |
|---|---:|---:|---:|
| MAE | \(0.257675\) | \(0.055593\) | \(78.43\%\) |
| RMSE | \(0.370737\) | \(0.083142\) | \(77.57\%\) |

Additional results:

$$
R^2_{\mathrm{test}} = 0.915416,
\qquad
\text{Improved fraction} = 88.39\%.
$$

The model generalizes well to the high-volatility regime.

---

### 8.2 High Monitoring Frequency Holdout

Holding out \(n=126\):

| Metric | Original TW | Corrected TW | Reduction |
|---|---:|---:|---:|
| MAE | \(0.083010\) | \(0.015735\) | \(81.04\%\) |
| RMSE | \(0.183500\) | \(0.021369\) | \(88.35\%\) |

Additional results:

$$
R^2_{\mathrm{test}} = 0.984449,
\qquad
\text{Improved fraction} = 60.00\%.
$$

The correction remains effective when the highest monitoring frequency is held out.

---

### 8.3 ATM Holdout

Holding out \(K=100\):

| Metric | Original TW | Corrected TW | Reduction |
|---|---:|---:|---:|
| MAE | \(0.090687\) | \(0.012768\) | \(85.92\%\) |
| RMSE | \(0.188914\) | \(0.015984\) | \(91.54\%\) |

Additional results:

$$
R^2_{\mathrm{test}} = 0.990697,
\qquad
\text{Improved fraction} = 63.75\%.
$$

The model performs strongly around the at-the-money region.

---

### 8.4 Long Maturity Holdout

Holding out \(T=2.0\):

| Metric | Original TW | Corrected TW | Reduction |
|---|---:|---:|---:|
| MAE | \(0.198503\) | \(0.077116\) | \(61.15\%\) |
| RMSE | \(0.325558\) | \(0.080484\) | \(75.28\%\) |

Additional results:

$$
R^2_{\mathrm{test}} = 0.909463,
\qquad
\text{Improved fraction} = 45.71\%.
$$

The correction still reduces error, but performance is weaker than in the interpolation tests. This suggests that long-maturity extrapolation is more challenging.

---

### 8.5 Leave-One-Moneyness Tests

The strongest results occur in the interior moneyness region.

For \(m=0.9\):

$$
\text{MAE reduction} = 88.07\%,
\qquad
\text{RMSE reduction} = 92.17\%,
\qquad
R^2_{\mathrm{test}} = 0.991392.
$$

For \(m=1.0\):

$$
\text{MAE reduction} = 85.92\%,
\qquad
\text{RMSE reduction} = 91.54\%,
\qquad
R^2_{\mathrm{test}} = 0.990697.
$$

For \(m=1.1\):

$$
\text{MAE reduction} = 75.58\%,
\qquad
\text{RMSE reduction} = 87.28\%,
\qquad
R^2_{\mathrm{test}} = 0.982341.
$$

However, performance is much weaker at the boundary moneyness levels.

For \(m=0.7\):

$$
\text{MAE reduction} = 10.76\%,
\qquad
R^2_{\mathrm{test}} = 0.790224,
\qquad
\text{Improved fraction} = 30.00\%.
$$

For \(m=1.3\):

$$
\text{MAE reduction} = 9.69\%,
\qquad
R^2_{\mathrm{test}} = 0.661273,
\qquad
\text{Improved fraction} = 38.75\%.
$$

This indicates that the model is strong at interpolation but weaker at boundary extrapolation.

When \(m=0.7\) or \(m=1.3\) is held out, the model has not seen data beyond that boundary, so it must extrapolate rather than interpolate.

---

## 9. Interpretation

The key finding is that the Turnbull-Wakeman approximation error is not random. It has a structured pattern across moneyness, volatility, maturity, and monitoring frequency.

The residual correction model is effective when predicting within the calibrated parameter grid. It substantially reduces MAE, RMSE, and maximum absolute error.

Strong performance is observed in:

- Random test split
- K-fold cross validation
- Checkerboard interpolation
- High volatility holdout
- High monitoring frequency holdout
- ATM and near-ATM moneyness regions

Weaker performance is observed in:

- Extreme moneyness boundary extrapolation
- Long maturity extrapolation

Therefore, the current model should be viewed as a strong interpolation-based correction method, not as a fully robust extrapolation engine.

---

## 10. Conservative Shrinkage Correction

A possible improvement is to apply a conservative shrinkage correction:

$$
C_{\mathrm{shrink}}
=
C_{\mathrm{TW}}
+
\lambda
\widehat{\varepsilon}_{\mathrm{TW}},
\qquad
0 \leq \lambda \leq 1.
$$

The original full correction corresponds to

$$
\lambda = 1.
$$

A smaller value, such as \(\lambda=0.5\) or \(\lambda=0.75\), may reduce overcorrection in regions where the Turnbull-Wakeman approximation is already accurate or where the model is extrapolating.

To avoid overfitting, \(\lambda\) should be selected on a validation set and then fixed across all robustness tests.

This can be used as a robustness diagnostic rather than as the main pricing model.

---

## 11. Current Conclusion

This project shows that a data-driven residual correction can significantly improve the Turnbull-Wakeman approximation for arithmetic Asian option pricing.

The main contribution is not to replace financial pricing theory with machine learning, but to combine:

$$
\text{Financial approximation}
+
\text{control variate benchmark}
+
\text{residual learning}.
$$

The corrected approximation keeps the speed advantage of analytical methods while moving closer to the accuracy of control variate Monte Carlo.

The strongest evidence comes from the checkerboard interpolation test, where the correction reduces MAE by approximately \(82\%\) and maintains out-of-sample \(R^2\) around \(0.985\).

The main limitation is boundary extrapolation, especially at extreme moneyness levels.

---

## 12. Future Work

Planned extensions include:

1. Expand the training grid to include more extreme moneyness levels:

$$
m \in \{0.6,0.7,\ldots,1.4\}.
$$

2. Add multiple spot levels:

$$
S_0 \in \{80,100,120\}.
$$

3. Compare full residual correction with conservative shrinkage correction.

4. Add residual heatmaps over moneyness and volatility.

5. Extend the framework to Greeks such as Delta, Gamma, Vega, Rho, and Theta.

6. Compare different residual models, such as polynomial Ridge regression, spline interpolation, Gaussian process regression, and gradient boosting.

7. Test the method under different market assumptions, such as stochastic volatility or local volatility.

---

## 13. Key Takeaway

The Turnbull-Wakeman approximation is fast but biased.

Control variate Monte Carlo is accurate but computationally expensive.

This project shows that the bias of the Turnbull-Wakeman approximation is learnable. By modeling the residual relative to a control-variate benchmark, the corrected approximation achieves much better accuracy while preserving the speed of an analytical approximation.

The method is especially effective for interpolation within a calibrated parameter grid, while extrapolation to extreme moneyness remains the main challenge.

# Asian Option Pricing Project

This project studies the pricing of Asian options under the Black–Scholes framework, with a focus on the contrast between **geometric-average** and **arithmetic-average** Asian options.

The main goal is to explain why geometric Asian options admit a closed-form pricing formula, while arithmetic Asian options generally require approximation methods or numerical simulation.

---

## Project Overview

Asian options are path-dependent derivatives whose payoff depends on the **average price** of the underlying asset over a monitoring period, rather than only on the terminal price at maturity.

This project is organized around the following core question:

> Why does the geometric-average Asian option have a closed-form solution under GBM, while the arithmetic-average Asian option generally does not?

To answer this, the project combines theory, approximation methods, Monte Carlo simulation, and a brief PDE extension.

---

## Core Structure

### 1. Definition of Asian Options
- Introduce Asian options and explain why they are **path dependent**
- Show how the payoff depends on the average of the underlying price over time
- Discuss practical motivation and applications

### 2. Arithmetic Average vs Geometric Average
- Define the **arithmetic average**
- Define the **geometric average**
- Explain the key difference:
  - the geometric average is analytically tractable under GBM
  - the arithmetic average is much harder because the average of lognormal variables is not lognormal

### 3. Geometric Asian Closed-Form Pricing
- Derive the distribution of the geometric average under GBM
- Show that the logarithm of the geometric average is normally distributed
- Obtain a Black–Scholes-type closed-form pricing formula
- Use this result as the theoretical benchmark

### 4. Arithmetic Asian Approximation Methods
- Explain why arithmetic Asian options do not generally have a simple closed form
- Implement and compare classical approximation methods:
  - **Turnbull–Wakeman approximation**
  - **Levy approximation**
- Discuss the intuition behind these approximation formulas

### 5. Monte Carlo Benchmark
- Simulate GBM paths under the risk-neutral measure
- Compute arithmetic-average Asian option payoffs path by path
- Estimate prices by Monte Carlo simulation
- Compare approximation methods in terms of:
  - pricing accuracy
  - stability
  - computational speed

### 6. PDE Extension: Večeř Approach
- Briefly explain why Asian option pricing naturally leads to a higher-dimensional problem
- Mention Večeř’s PDE framework as a more advanced and systematic approach
- Position it as a higher-level extension beyond approximation and Monte Carlo

---

## Mathematical Setup

Under the risk-neutral measure, the underlying asset price is assumed to follow geometric Brownian motion:

\[
dS_t = r S_t\,dt + \sigma S_t\,dW_t
\]

where:
- \(S_t\): asset price
- \(r\): risk-free rate
- \(\sigma\): volatility
- \(W_t\): Brownian motion

For discretely monitored Asian options, the averages are typically defined as:

### Arithmetic Average
\[
A_n^{(a)} = \frac{1}{n}\sum_{i=1}^n S_{t_i}
\]

### Geometric Average
\[
A_n^{(g)} = \left(\prod_{i=1}^n S_{t_i}\right)^{1/n}
\]

---

## Main Objectives

The main objectives of this project are:

- understand why Asian options are path dependent
- explain the difference between arithmetic and geometric averaging
- derive and implement the closed-form price for geometric Asian options
- implement approximation methods for arithmetic Asian options
- use Monte Carlo simulation as a benchmark
- compare methods in terms of accuracy and efficiency
- briefly discuss PDE-based extensions

---

## Methods Included

This project includes or plans to include:

- **Geometric Asian closed-form pricing**
- **Arithmetic Asian Monte Carlo pricing**
- **Turnbull–Wakeman approximation**
- **Levy approximation**
- Optional variance reduction with **control variates**
- Brief discussion of **Večeř PDE approach**

---

## Numerical Experiments

The numerical part of the project will compare methods across different parameter settings, such as:

- strike \(K\)
- volatility \(\sigma\)
- maturity \(T\)
- number of monitoring dates \(n\)

Typical outputs include:

- price comparison tables
- approximation error relative to Monte Carlo
- runtime comparison
- sensitivity plots

---

## Expected Comparison Table

| Method | Price | Std Error | Abs Error vs MC | Runtime |
|--------|------:|----------:|----------------:|--------:|
| Geometric Closed Form | ... | ... | ... | ... |
| Turnbull–Wakeman | ... | ... | ... | ... |
| Levy | ... | ... | ... | ... |
| Monte Carlo | ... | ... | 0 | ... |

---

## Main Takeaway

The central message of this project is:

> The geometric-average Asian option is analytically tractable because the logarithm of the geometric average remains normally distributed under GBM.  
> In contrast, the arithmetic-average Asian option is much harder because the arithmetic average of lognormal prices is not itself lognormal, which motivates the use of approximation methods, Monte Carlo simulation, and PDE approaches.

---

## Repository Structure

A possible file organization is:

```text
AsianOption/
│
├── README.md
├── geometric_asian.py (done)
├── arithmetic_asian_mc.py (done)
├── turnbull_wakeman.py
├── levy_approximation.py
├── control_variate.py
├── experiments.py
└── report/
```

## Future improvement (my suggestion, not sure if we can make it)
- Any other control variable?
- quasi-monte carlo method?
- what about the pricing under stochastic vol?
- Greek? how to hedge?
- consider the approximation error under different method
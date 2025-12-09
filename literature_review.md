# Literature Review and Background

## 1. Problem Setting

Our project studies **time-series forecasting of NVIDIA (NVDA) stock prices**. Stock data are:

- noisy and highly volatile
- strongly affected by market-wide conditions (SPY, VIX)
- potentially nonstationary due to structural changes (e.g., AI boom, macro shocks)

This makes NVDA a good test case for comparing **simple baselines** (naïve/random-walk) and a modern **additive model** (Prophet) with exogenous regressors.

We use **daily historical data** for NVDA (adjusted close and volume), SPY (S&P 500 ETF), and VIX (CBOE Volatility Index), downloaded from Yahoo Finance and aligned by date.

---

## 2. Classical Forecasting Concepts

Classical time-series forecasting methods such as **ARIMA, exponential smoothing, and state-space models** are well established in the forecasting literature. Hyndman & Athanasopoulos emphasize three core ideas:

1. Always compare against simple benchmarks such as:
   - **Naïve / random walk**: \(\hat{y}_{t+1} = y_t\)
   - **Mean forecast** over the training window
2. Decompose series into **trend**, **seasonality**, and **remainder**.
3. Evaluate models on **held-out future data** using appropriate error metrics (MAE, RMSE, MAPE, \(R^2\)).  

These ideas motivate our project design:
- We treat the naïve model as a strong baseline.
- We explicitly model trend and seasonality via Prophet.
- We evaluate on a chronologically later test set instead of random shuffles.

---

## 3. Prophet: Additive Models for Forecasting at Scale

**Prophet**, developed at Facebook (Meta), is a forecasting tool for time series with:

- **Non-linear trend** with automatic changepoints.
- **Multiple seasonal components** (yearly, weekly, daily).
- **Holiday / event effects**.
- **Extra user-defined regressors** (e.g., SPY, VIX, volume).  

The core idea is a **decomposable additive model**:

\[
y(t) = g(t) + s(t) + h(t) + \epsilon_t
\]

where:
- \(g(t)\) is the trend component,
- \(s(t)\) is seasonal structure,
- \(h(t)\) captures holidays/regressors,
- \(\epsilon_t\) is noise.

Taylor & Letham highlight several properties that make Prophet attractive for “forecasting at scale”:

- It is **robust to missing data and outliers**.
- It allows **analysts to configure** changepoints, seasonalities, and regressors without deep time-series expertise.
- It produces **interpretable components** (plots of trend, seasonality, regressor effects) rather than a black box. :contentReference[oaicite:0]{index=0}

In our project, Prophet is used as the main model because:

- NVDA exhibits **strong long-term trend shifts**.
- We can inject **market context** via regressors (SPY, VIX, volume).
- We want interpretable, component-wise plots rather than only raw predictions.

---

## 4. Extensions and Alternatives

### 4.1 NeuralProphet

**NeuralProphet** is a hybrid framework that builds on Prophet’s additive structure but adds:

- auto-regressive terms,
- covariate modules, and
- optional neural network components

implemented in PyTorch. It keeps the Prophet-style decomposable model but introduces **local context** via autoregression, which can significantly improve short- and medium-horizon accuracy on some datasets. :contentReference[oaicite:1]{index=1}

We did not implement NeuralProphet in this project (due to scope and time), but it is a natural next step in future work.

### 4.2 Deep Learning Models (LSTMs, Transformers)

There is a large literature on using **LSTMs, GRUs, and Transformer-based architectures** for stock prediction and general time-series forecasting. These models:

- can capture nonlinear dynamics and longer temporal dependencies,
- often need **more data, careful tuning, and regularization**, and
- are less interpretable than Prophet-style models.

For an EE344 course project, we prioritized:

- a fully working, interpretable pipeline (Prophet + regressors + rolling evaluation),
- plus a strong naïve baseline,

rather than a partially tuned deep model.

---

## 5. Use of Exogenous Regressors in Financial Forecasting

In financial modeling, it is common to include **market factors** and **risk measures** as regressors:

- broad market index (e.g., SPY) to represent market movement,
- volatility index (VIX) to capture “fear” or risk sentiment,
- trading volume as a proxy for liquidity and attention.

Prophet natively supports **additional regressors**, making it easy to integrate these factors: they are treated as extra columns and their effects are estimated as part of the additive model. :contentReference[oaicite:2]{index=2}

This motivates our final model design:
- target: log-transformed NVDA price,
- regressors: log-transformed SPY, VIX, and NVDA volume.

---

## 6. Summary of Takeaways from the Literature

From the literature we reviewed, the main guiding principles were:

1. **Baselines matter**: naïve/random-walk forecasts are surprisingly hard to beat in stock data; any advanced model should be compared against them.
2. **Decomposition helps**: separating trend, seasonality, and regressors improves interpretability and often robustness.
3. **Exogenous information is valuable**: macro-market indices and volatility measures add information that single-stock history alone cannot capture.
4. **Interpretability vs. complexity**: Prophet offers a good balance for an academic project—more flexible than pure ARIMA, but simpler and more transparent than deep neural networks.
5. **Rolling evaluation is closer to reality**: evaluating models in a rolling or walk-forward manner approximates how forecasts would be generated in practice.

---

## 7. References

- Taylor, S. J., & Letham, B. (2018). *Forecasting at Scale*. The American Statistician, 72(1), 37–45.  
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.  
- Triebe, O., Hewamalage, H., Pilyugina, P., Laptev, N., Bergmeir, C., & Rajagopal, R. (2021). *NeuralProphet: Explainable Forecasting at Scale* (arXiv:2111.15397).  
- Meta Open Source. *Prophet: Forecasting at Scale* – Official documentation.  

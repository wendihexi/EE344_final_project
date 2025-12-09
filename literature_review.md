# Literature Review: Time-Series Forecasting for NVDA Stock

## 1. Problem Setting

Our project focuses on **forecasting NVIDIA (NVDA) daily closing prices**. Equity time series are:

- noisy and highly volatile,
- nonstationary (regime shifts, bubbles, crashes),
- affected by broader market conditions (market indices, volatility, news).

Because of this, NVDA is a good testbed for comparing:

- **Simple statistical baselines** (naïve/persistence),
- **Additive forecasting models** (Prophet),
- **Deep sequence models** (LSTMs).

Both versions of our code use **daily NVDA data from Yahoo Finance** (`yfinance`) and treat forecasting as a supervised time-series problem.

---

## 2. Classical Forecasting Background and Baselines

Classical methods such as **ARIMA, exponential smoothing, and state-space models** are the backbone of time-series forecasting. Hyndman & Athanasopoulos emphasize three key ideas:​:contentReference[oaicite:0]{index=0}  

1. Always compare new models against **simple baselines**, especially:
   - **Naïve / random-walk**: \(\hat{y}_{t+1} = y_t\),
   - **Mean forecast** over the training window.
2. Explicitly separate **trend**, **seasonality**, and **residuals**.
3. Evaluate on **future hold-out data** (not shuffled) with metrics like MAE, RMSE, and \(R^2\).

In **v1**, we followed this advice by building a **naïve persistence model** and using it as a strong benchmark against Prophet and the LSTM. The results in v1 show that for short horizons on NVDA, the naïve model already achieves a very high \(R^2\) (~0.95), which matches the literature’s warning that financial series are often close to random walks.

In **v2**, we kept the same spirit: Prophet variants are always evaluated against a simple baseline (naïve and a straightforward “baseline Prophet” without extra tricks).

---

## 3. Prophet: Additive, Interpretable Models

**Prophet** (Facebook/Meta) is a decomposable time-series model with:

- non-linear trend with changepoints,
- multiple seasonal components (weekly, yearly, etc.),
- holiday / event effects,
- optional extra regressors.:contentReference[oaicite:1]{index=1}  

Taylor & Letham describe Prophet as a practical tool for **“forecasting at scale”**: it provides a modular regression model where analysts can tune trend, seasonality, and holiday components without being deep time-series experts.:contentReference[oaicite:2]{index=2}  

In our project:

- **v1**:
  - Used Prophet on NVDA close price.
  - Experimented with **US holidays** and custom **NVDA “AI event” holidays** (GTC and other major AI-related announcements).
  - Extended holiday effects via `lower_window` and `upper_window` to model multi-day impacts.
- **v2**:
  - Reorganized Prophet usage into a **clean pipeline**:
    - Baseline Prophet on price,
    - Prophet on **log price with flexible trend**,
    - Prophet with **explicit seasonality**,
    - **Rolling Prophet** (one-step-ahead) with and without **exogenous regressors**:
      - SPY (market index),
      - NVDA trading volume,
      - VIX (volatility index).

This aligns with Prophet’s design: incorporate domain knowledge via holidays / regressors and allow trends to adapt to structural changes.:contentReference[oaicite:3]{index=3}  

---

## 4. Deep Learning for Stock Prediction: LSTMs

**Long Short-Term Memory (LSTM)** networks are widely used for sequence modeling. They can capture non-linear temporal dependencies and have been applied to stock prediction with success:

- Fischer & Krauss use LSTMs to predict directional movements of S&P 500 constituents and show that LSTMs can outperform memory-free models like random forests, standard dense nets, and logistic regression on daily returns.:contentReference[oaicite:4]{index=4}  
- Subsequent work and surveys report strong performance of LSTMs and related architectures on various stock markets and indices.:contentReference[oaicite:5]{index=5}  

In **v1**, we implemented a **pure LSTM pipeline**:

- Scale NVDA close prices with MinMaxScaler,
- Create sliding-window sequences (e.g., 60 days → next-day price),
- Train a two-layer LSTM with 50 units per layer for 50 epochs,
- Evaluate on a held-out test split.

The LSTM achieved:

- **RMSE ≈ 3.33**  
- **MAE ≈ 1.96**  
- **R² ≈ 0.9964**  

on the (scaled back) test set, indicating an extremely tight fit to the test data.

The literature warns that such high in-sample or single-split performance on financial data can reflect **overfitting** or evaluation choices (e.g., lack of rolling evaluation, using overlapping windows, or not testing on truly “future” periods). Our results are consistent with LSTMs’ ability to fit complex patterns, but they need cautious interpretation.

---

## 5. Hybrid and Successor Models (NeuralProphet)

Recent work proposes **NeuralProphet**, a hybrid framework that combines Prophet-style additive components with neural auto-regression and covariate modules in PyTorch.:contentReference[oaicite:6]{index=6}  

Key points from NeuralProphet:

- Retains Prophet’s **trend/seasonality/holiday** decomposition.
- Adds **local context** via autoregressive and neural components.
- Aims to bridge the gap between **interpretable additive models** and **flexible deep learning**.

We did not implement NeuralProphet in either version, but it strongly motivates our design:

- **v1** explores a *deep* model (LSTM) separately from Prophet.
- **v2** focuses on making **Prophet as strong and well-engineered as possible**, using:
  - log transforms,
  - additional regressors (SPY, VIX, volume),
  - rolling evaluation to mimic realistic deployment.

In future work, NeuralProphet would be a natural candidate model that combines the strengths of both versions.

---

## 6. How the Literature Informed Both Versions

- From **classical forecasting literature**, we took:
  - The importance of **naïve benchmarks** and **proper train/test splitting**.
  - The idea of modeling trend/seasonality explicitly.:contentReference[oaicite:7]{index=7}  

- From the **Prophet paper**, we took:
  - Use of **changepoints**, **holiday events**, and **extra regressors**.
  - Focus on **interpretable components**, not just accuracy.:contentReference[oaicite:8]{index=8}  

- From **LSTM stock forecasting work**, we took:
  - The motivation to try sequence models on NVDA.
  - Awareness of the risk of overfitting and the need for robust evaluation.:contentReference[oaicite:9]{index=9}  

**v1** can be seen as “broad exploration”: naïve vs Prophet vs LSTM, showing how a deep model can fit NVDA extremely well on a single split while Prophet struggles without careful engineering.

**v2** is the “tightened” version: it focuses on Prophet and preprocessing, builds a cleaner pipeline with exogenous regressors and rolling evaluation, and sticks closer to the course’s emphasis on interpretable models and method comparison.

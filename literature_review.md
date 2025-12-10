# Literature Review: Time-Series Forecasting for NVDA Stock

## 1. Problem Setting

Our project focuses on forecasting **NVIDIA (NVDA) daily closing prices**. Equity time series are:

- noisy and highly volatile,  
- nonstationary (regime shifts, bubbles, crashes),  
- affected by broader market conditions (indices, volatility, news).  

Because of this, NVDA is a useful testbed for comparing:

- **simple statistical baselines** (naïve / persistence),  
- **additive forecasting models** (Prophet),  
- and, in our initial version, **deep sequence models** (LSTMs).

Both versions of our code use **daily NVDA data from Yahoo Finance** (via `yfinance`) and treat forecasting as a supervised time-series problem.

---

## 2. Classical Forecasting Background and Baselines

Classical methods such as ARIMA, exponential smoothing, and state-space models form the backbone of time-series forecasting (Hyndman & Athanasopoulos, 2018). Three ideas from this literature guide our project:

1. Always compare new models against **simple baselines**, especially:
   - the **naïve / random-walk** predictor
     $$
      \hat{y}_{t+1} = y_t.
     $$
   - and simple mean forecasts.  
2. Explicitly consider **trend**, **seasonality**, and **residuals** rather than treating the series as a black box.  
3. Evaluate on **future hold-out data** (not shuffled) using metrics like MAE, RMSE, and \(R^2\).

In **Version 1 (v1)**, we followed this advice by building a naïve persistence model and using it as a strong benchmark against Prophet and the LSTM. For short horizons on NVDA, the naïve model achieves a very high \(R^2\), which aligns with the common result that financial series often resemble random walks. In **Version 2 (v2)**, we keep the same philosophy: every Prophet variant is evaluated against the naïve baseline and a straightforward “baseline Prophet” model.

---

## 3. Prophet: Additive, Interpretable Models

**Prophet** (Taylor & Letham, 2018) is a decomposable time-series model with:

- non-linear trends with changepoints,  
- multiple seasonal components (e.g., weekly, yearly),  
- holiday or event effects,  
- and optional extra regressors.  

It is designed as a practical tool for “forecasting at scale,” where analysts can tune trend, seasonality, and holiday components without deep time-series expertise. The model structure is:

$$
y(t) = g(t) + s(t) + h(t) + \sum_j \beta_j x_j(t) + \varepsilon_t
$$


where \(g(t)\) is the trend, \(s(t)\) seasonal effects, \(h(t)\) holiday or event effects, and \(x_j(t)\) are user-defined regressors.

In our project:

- **v1** uses Prophet on NVDA close price, experiments with US holidays and custom “AI event” dates (e.g., GTC keynotes), and extends holiday effects over multi-day windows.  
- **v2** reorganizes Prophet into a cleaner pipeline:
  - baseline Prophet on raw price,  
  - Prophet on **log price** with a more flexible trend,  
  - Prophet with explicit weekly/yearly seasonality,  
  - and **rolling Prophet** (one-step-ahead) with and without exogenous regressors such as SPY (market index), NVDA volume, and VIX (volatility index).

These choices follow Prophet’s intended use: incorporate domain knowledge through holidays and regressors, and allow the trend to adapt to structural changes like the AI boom.

---

## 4. Deep Learning for Stock Prediction: LSTMs (v1)

Long Short-Term Memory (LSTM) networks are widely used for sequence modeling and have been applied to stock prediction. Fischer and Krauss (2018), for example, show that LSTMs can outperform memory-free models such as random forests and logistic regression when predicting directional movements of S&P 500 constituents. Other studies and surveys report strong performance of LSTMs and related architectures on diverse financial markets.

In **v1**, we implemented a pure LSTM pipeline:

- scale NVDA close prices with MinMaxScaler,  
- create sliding-window sequences (e.g., 60 days → next-day price),  
- train a two-layer LSTM,  
- and evaluate on a held-out test split.

The LSTM achieved an \(R^2\) close to 1 on that split, indicating an extremely tight fit. The literature, however, warns that very high performance on a single split for financial data can reflect **overfitting** or optimistic evaluation (e.g., overlapping windows, limited test periods). This guided our later decision in v2 to focus more on Prophet, baselines, and evaluation protocol rather than deep models.

---

## 5. Hybrid and Successor Models (NeuralProphet)

More recent work proposes **NeuralProphet**, a hybrid framework that combines Prophet-style additive components with neural autoregression and covariate modules implemented in PyTorch (Triebe et al., 2021). It keeps Prophet’s interpretable decomposition (trend/seasonality/holidays) but adds:

- local context through autoregressive terms,  
- and flexible non-linear components via neural networks.

We did not implement NeuralProphet in either version, but it motivates our design:

- **v1** explores a deep model (LSTM) separately from Prophet.  
- **v2** focuses on making **Prophet as strong and well-engineered as possible**, using log transforms, additional regressors (SPY, VIX, volume), and **rolling evaluation** to better mimic real-world deployment.

NeuralProphet is a natural candidate for future work that could combine the interpretability of Prophet with the flexibility of LSTMs.

---

## 6. How the Literature Informed Both Versions

From the literature, we adopted several guiding principles:

- From **classical forecasting** (Hyndman & Athanasopoulos, 2018):  
  - always include **naïve benchmarks** and proper **time-ordered train/test splits**;  
  - think in terms of trend/seasonality rather than pure black-box fitting.

- From the **Prophet** work (Taylor & Letham, 2018; Prophet documentation):  
  - use **changepoints**, holiday effects, and extra regressors to encode domain knowledge;  
  - value **interpretability** (trend and component plots) alongside accuracy.

- From **LSTM stock-forecasting studies** (Fischer & Krauss, 2018 and others):  
  - recognize the potential of deep sequence models,  
  - but also the risk of overfitting and the need for robust evaluation protocols.

Put together, **v1** is a broad exploration—naïve vs. Prophet vs. LSTM—showing how a deep model can fit NVDA extremely well on a single split while Prophet struggles without careful engineering. **v2** tightens the design: it emphasizes Prophet and preprocessing, builds a cleaner pipeline with exogenous regressors and rolling evaluation, and stays closer to the course’s emphasis on interpretable models and method comparison.

---

## References

- Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *European Journal of Operational Research, 270*(2), 654–669.  
- Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice* (2nd ed.). OTexts.  
- Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American Statistician, 72*(1), 37–45.  
- Triebe, O., Hewamalage, H., Pilyugina, P., Laptev, N., Bergmeir, C., & Rajagopal, R. (2021). NeuralProphet: Explainable forecasting at scale. *arXiv preprint* arXiv:2111.15397.  
- Meta Open Source. *Prophet: Forecasting at scale* — official documentation.  

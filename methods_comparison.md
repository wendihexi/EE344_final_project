# Methods and Model Comparison

## 1. Overview of Models

In this project we compared several forecasting setups on daily NVIDIA (NVDA) stock data:

1. **Baseline Prophet model**
   - Input: NVDA adjusted close price.
   - No log transform, default trend and seasonality settings.
   - No additional regressors.

2. **Improved Prophet: log-transformed price + flexible trend**
   - Target `y` = log(adjusted close), which stabilizes variance and reduces scale issues.
   - Increased changepoint flexibility to allow the trend to adapt to large regime shifts.
   - Predictions are exponentiated back to the price scale before evaluation.

3. **Enhanced Prophet with exogenous regressors (final model)**
   - Same log-transformed target as the improved model.
   - Adds extra regressors:
     - SPY (broad market index) as a market-level factor.
     - NVDA trading volume as a proxy for liquidity and attention.
     - VIX (volatility index) as a measure of market fear/uncertainty.
   - Trained in a rolling-window fashion so that each forecast uses only past data.

We also describe a **naïve baseline** conceptually (forecast tomorrow’s price as today’s price), which is a standard benchmark in financial time-series forecasting, even though the full implementation is not central in this notebook.

## 2. Data Splits and Evaluation

- The dataset is split chronologically into **train** and **test** sets.
- Models are fit on the training period and evaluated on the held-out test period.
- Our main evaluation metric is **R² score** on the test set, computed on the original (non-log) price scale.

This setup mirrors standard practice in time-series forecasting, where we respect temporal order and evaluate on future data.

## 3. Baseline Prophet

The baseline Prophet model is intentionally simple:

- Uses default yearly/weekly seasonality assumptions.
- No log transform and no extra regressors.
- Single fit over the training window followed by a forecast over the test window.

On NVDA data, this baseline Prophet model produced a **negative R²** on the test set, indicating that it performed worse than just predicting the mean. This highlights that:

- Raw NVDA price is highly volatile and nonstationary.
- Default Prophet settings are not sufficient for this stock without further engineering.

## 4. Log-Transformed Prophet

To address variance and scale issues, we transform the target:

- Let `y = log(price)`.
- Fit Prophet on `y`.
- Exponentiate the predictions back to obtain prices.

This stabilizes variance and reduces the impact of large outliers, which generally improves the ability of the model to capture trend. In our experiments, the log-transform reduced overfitting and improved visual alignment between predicted and actual prices, although the R² gains were still limited without additional context features.

## 5. Prophet with SPY, Volume, and VIX Regressors

The final and best-performing model adds external regressors:

- `add_regressor("spy")` — captures broad market moves.
- `add_regressor("volume")` — captures liquidity and attention spikes.
- `add_regressor("vix")` — captures changes in risk sentiment and volatility.

We train this model in a **rolling window** configuration:

- Slide a training window forward through time.
- Refit Prophet on each window using past data and regressors.
- Generate a short-horizon forecast and compare against the next test observations.

This setup more closely mimics how a real trader would update forecasts over time and helps reduce look-ahead bias.

Empirically:

- The **baseline Prophet** showed very poor out-of-sample performance (R² < 0).
- The **log-transformed + regressor model** improved the R² noticeably (our notebook prints the R² for the rolling Prophet with regressors).
- Visual inspection of the forecast vs actual series shows that the final model better tracks major upward and downward movements in NVDA, especially when large market shocks coincide with VIX spikes.

(You can insert your actual numeric R² values from the notebook in this section when you finalize the report.)

## 6. When Each Method Works Well / Poorly

- **Naïve / random-walk forecast**
  - Works well:
    - For very short horizons on highly efficient markets where price changes are close to random.
  - Fails:
    - When there are strong, predictable seasonal or trend components.

- **Baseline Prophet**
  - Works well:
    - On business or demand time series with clear seasonality, smoother trends, and moderate noise.
  - Fails:
    - On raw NVDA prices without transform or regressors, where volatility and regime shifts dominate.

- **Log-Prophet + Regressors (our final model)**
  - Works well:
    - When external variables (SPY, VIX, volume) explain some of the variance in NVDA.
    - For capturing trend and medium-term structure rather than tick-by-tick noise.
  - Still limited:
    - In truly chaotic periods (earnings surprises, extreme macro events) where price jumps are driven by information not present in the regressors.
    - Over very long horizons where model assumptions may no longer hold.

## 7. Design Decisions Linked to Literature

Our final modeling choices are directly motivated by the literature:

- Using an additive, decomposable model (Prophet) for interpretability and robustness.
- Treating naïve/random-walk as a baseline benchmark, following standard forecasting practice.
- Incorporating exogenous regressors that represent market conditions (SPY), volatility (VIX), and liquidity (volume), which is consistent with financial forecasting research that emphasizes multi-factor models.
- Using a rolling evaluation scheme to better approximate realistic deployment and avoid optimistic estimates.

These decisions helped turn a poorly performing baseline Prophet model into a more competitive forecasting pipeline for NVDA.

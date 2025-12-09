# Methods, Preprocessing, and Model Comparison

## 1. Dataset and Overall Pipeline

We worked with **daily historical data** for:

- NVIDIA (NVDA): adjusted close and volume
- SPY: closing price (broad US market proxy)
- VIX: closing value (market volatility index)

All series are downloaded from Yahoo Finance, merged on the calendar date, and restricted to days where NVDA is traded. The high-level pipeline is:

1. Download & merge NVDA, SPY, VIX.
2. Clean and align dates (single `ds` column).
3. Apply log transforms to stabilize variance.
4. Split chronologically into train/test sets.
5. Train and compare:
   - naïve baseline,
   - baseline Prophet,
   - improved Prophet (log + regressors).
6. Evaluate forecasting accuracy on the test window.

---

## 2. Preprocessing Steps

### 2.1 Column Cleaning and Date Handling

Yahoo Finance returns a DataFrame with a datetime index and multiple columns. We:

- Flattened the columns (if MultiIndex) and reset the index.
- Kept only the needed columns (`Date`, `Close`, `Volume`).
- Renamed to Prophet’s expected format:

  - `ds` = date
  - `y`  = target (NVDA adjusted close)

- Ensured `ds` is parsed as a proper `datetime` type.

This makes the data compatible with Prophet’s API and ensures consistent time handling.

### 2.2 Creating and Merging Regressors

We construct three exogenous regressors:

- `SPY` (market level)
- `Volume` (NVDA trading volume)
- `VIX` (volatility / risk sentiment)

Steps:

1. Download SPY and VIX as separate time series.
2. Reset each to have a `Date` column and rename:
   - `Close → SPY`
   - `Close → VIX`
3. Extract NVDA `Volume` as a separate series.
4. Merge all three onto the main NVDA frame on the `ds` column using left joins.
5. Handle missing values with **forward fill** to keep the regressors defined for all trading days.

### 2.3 Log Transformation

To stabilize variance and reduce the impact of large outliers, we take logs:

- `y      = log(NVDA price)`
- `SPY    = log(SPY close)`
- `Volume = log(Volume)`
- `VIX    = log(VIX)`

This is a standard practice for financial data, where price and volume can grow exponentially over long horizons.

### 2.4 Train/Test Split and Rolling Setup

We split the dataset chronologically:

- **Training set**: early part of the time series.
- **Test set**: later unseen period.

For the final model we use a **rolling window** strategy:

1. Initialize a training window with early data.
2. Fit Prophet on the current training window.
3. Forecast one (or a small number of) future point(s).
4. Compare prediction with actual values.
5. Extend the training window to include the new true observation.
6. Repeat through the test period.

This mimics a realistic deployment where only past data are available at each forecast step and reduces look-ahead bias.

---

## 3. Models Considered

### 3.1 Naïve Baseline (Random Walk)

**Definition:**

\[
\hat{y}_{t+1} = y_t
\]

This forecast simply predicts that tomorrow’s price equals today’s price.

- **Why use it?**
  - In efficient markets, short-term price changes are close to random; the random-walk is a strong benchmark.
  - It is trivial to implement and interpret.

- **When it works well**
  - Over very short horizons for highly efficient financial assets.
  - When there is no strong predictable structure beyond noise.

- **When it fails**
  - When there are strong trends or seasonal patterns that persist beyond one day.
  - When the interest is in longer-horizon forecasts.

---

### 3.2 Baseline Prophet (Price Only)

We first fit a simple Prophet model on the **raw NVDA price** (no log, no regressors):

- Default yearly/weekly seasonality.
- Automatic changepoints for trend.
- Single fit over the train window; forecast across the entire test window.

**Findings:**

- The model **visually fits** major long-term trends.
- However, out-of-sample performance (e.g., \(R^2\) on the test set) is poor and can be **worse than the naïve baseline**.
- The model struggles with:
  - high volatility,
  - abrupt structural breaks,
  - heteroskedasticity (variance increasing over time).

This motivated further preprocessing and feature engineering.

---

### 3.3 Prophet with Log-Transformed Target

Next, we fit Prophet to **log price**:

- Replace `y` by `log(price)` before fitting.
- After forecasting log values, exponentiate predictions to return to price scale.

**Why log-transform?**

- Reduces scale variation and stabilizes variance.
- Treats proportional changes (returns) more consistently over time.
- Often improves the ability of models to capture trend.

**Effect:**

- The log-transform improves stability and reduces extreme residuals.
- Visual fit in the test region looks more reasonable, but performance is still limited with only NVDA’s own price as input.

---

### 3.4 Final Model: Prophet + Log Transform + Regressors + Rolling Evaluation

Our final and best-performing configuration is:

- **Target:** `y = log(NVDA price)`
- **Regressors:**
  - `SPY` (log),
  - `Volume` (log),
  - `VIX` (log)
- **Model:** Prophet with `add_regressor` for each of the above.
- **Evaluation:** Rolling-window forecasting on the test period.

**Rationale:**

- SPY adds *market-level* information.
- VIX adds *risk and volatility* information.
- Volume adds *attention/liquidity* information.
- The rolling setup keeps the model updated with the latest data and avoids using any future information.

**Observed behavior (qualitative):**

- The final model follows major upward and downward movements of NVDA more closely than the baseline Prophet.
- It yields a **higher \(R^2\) on the test set** compared to the earlier Prophet variants.
- It remains imperfect around sharp earnings jumps or unexpected macro events, which are difficult to predict from past prices and the chosen regressors alone.

(When finalizing the report, you can insert the exact numeric values of MAE / RMSE / \(R^2\) from your notebook here.)

---

## 4. Methodological Comparison

| Aspect                      | Naïve Baseline           | Baseline Prophet                  | Log Prophet + Regressors (Final)         |
|-----------------------------|--------------------------|------------------------------------|------------------------------------------|
| Inputs                      | NVDA price only          | NVDA price only                   | NVDA price + SPY + Volume + VIX          |
| Transform                   | none                     | none                               | log-transform on target + regressors     |
| Complexity                  | very low                 | moderate                           | higher but still manageable              |
| Interpretability            | trivial                  | good (trend/seasonality plots)    | good (trend + regressor effects)         |
| Short-horizon performance   | surprisingly strong      | mixed                              | better tracking of movements             |
| Long-horizon performance    | poor                     | captures trend but misses shocks   | better medium-term structure             |
| Robustness to volatility    | none                     | limited                            | improved via log-transform + regressors  |

---

## 5. What Worked, What Didn’t, and Why

### What Worked Well

- **Log transform** helped stabilize variance and improved fit.
- **Exogenous regressors (SPY, VIX, Volume)** captured some variation that pure NVDA history could not explain.
- **Rolling evaluation** gave a more realistic sense of performance and avoided overly optimistic metrics.

### What Didn’t Work as Well

- A single global Prophet fit on raw prices over the entire history performed poorly on the test set.
- Prophet still struggled with **sudden jumps** (earnings surprises, news shocks) that are not encoded in regressors.

### Why We Chose This Approach

- Prophet directly supports **additive decomposition and extra regressors**, matching both the literature and the nature of our data.
- The model is **interpretable** and can be easily tuned by adjusting changepoint priors, seasonalities, and regressors.
- For an EE344 project, it offers a strong balance between:
  - theoretical grounding (backed by published literature),
  - practical implementation,
  - and explainable results.

---

## 6. Limitations and Future Work

- **Additional features:** News sentiment, options data, sector indices, or macroeconomic variables could further improve forecasts.
- **Alternative models:** 
  - NeuralProphet (hybrid neural/additive model),
  - LSTMs or Transformers for more complex, nonlinear dynamics.
- **Risk-aware metrics:** Exploring risk-focused measures (e.g., downside error, quantile loss) could give a more nuanced picture than MSE-based metrics alone.

Despite these limitations, our final Prophet configuration demonstrates how preprocessing (log transforms and feature engineering) and method choice (additive model with regressors, rolling evaluation) can significantly improve performance over both a naïve baseline and an untuned baseline Prophet on NVDA stock data.

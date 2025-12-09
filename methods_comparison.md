# Methods, Preprocessing, and Comparison of Code Versions

We have two main versions of our project code:

- **Version 1 (v1)**: `EE_344_Final_project-4.ipynb`  
  – Original final-project notebook combining naïve baseline, Prophet, and LSTM.

- **Version 2 (v2)**: `Improved_ver_of_A's_EE344_Final_temp1.ipynb`  
  – Improved, Prophet-centric notebook with cleaner preprocessing and rolling evaluation.

This document describes the **shared pipeline**, what we did in **each version**, key **results**, and **why** we made these choices.

---

## 1. Data and Shared Preprocessing

Both versions use **daily NVDA stock data** from Yahoo Finance via `yfinance`:

- Download from `1999-01-01`.
- Extract and rename:
  - `ds` = date,
  - `y`  = target (NVDA adjusted/close price).

In both:

- The date column is converted to `datetime`.
- Data are sorted chronologically.
- For Prophet, we ensure the DataFrame has exactly `['ds', 'y']` with float `y`.

### v2-Only: Extra Market Regressors

In v2, we add additional daily series:

- **SPY** (S&P 500 ETF close),
- **VIX** (volatility index),
- **Volume** (NVDA trading volume).

These are:

1. Downloaded with `yfinance`,
2. Reset to `(ds, value)` format,
3. Merged onto the NVDA frame via left joins on `ds`,
4. Forward-filled to handle occasional missing days,
5. Log-transformed along with `y`:

   - `log_price = log(NVDA Close)`,
   - `log_SPY`, `log_VIX`, `log_Volume`.

This yields a clean DataFrame for Prophet with exogenous regressors.

---

## 2. Version 1 (Original): Naïve, Prophet, and LSTM

### 2.1 Naïve (Persistence) Baseline

In v1 we implement a **persistence model**:

\[
\hat{y}_{t+1} = y_t
\]

Procedure:

- Use a **recent train window (≈2 years)** and **later test window**.
- For 1-day-ahead forecasts, take today’s close as tomorrow’s prediction.

**Results (1-day ahead, recent period):**

- **MAE ≈ 1.23**
- **RMSE ≈ 1.70**
- **R² ≈ 0.95**

This confirms that for very short horizons on NVDA, simply repeating yesterday’s price is already extremely strong.

### 2.2 Prophet in v1: Holidays and AI Events

We then apply Prophet to NVDA:

1. **Baseline Prophet**:
   - Trend + default seasonalities.
   - Train on the same train window, forecast 1 day ahead in the test window.
   - Result: Prophet performs much worse than naïve (R² ≈ −76), indicating a major mismatch between default Prophet behavior and the NVDA series in this setup.

2. **Holidays & AI Events**:
   - Add standard US holidays via `Prophet(holidays=country_holidays("US"))`.
   - Define a custom `nvda_ai_holidays` DataFrame with dates of major NVIDIA AI-related announcements (e.g., GTC keynotes, architecture launches).
   - Add `lower_window = 0` and `upper_window = 6` so each event influences a 7-day window, modeling the idea that AI events affect NVDA for about a week.

Qualitatively, inspection of the **components plots** shows:

- Holiday effects capturing short bursts around AI events,
- Some localized adjustments to the trend around these dates.

However, **quantitative improvement vs naïve is minimal** in v1’s current Prophet configuration; the main story is that naïve remains superior.

### 2.3 LSTM Model in v1

To explore deep learning, v1 builds a **two-layer LSTM**:

1. **Scaling**:
   - Use `MinMaxScaler(0, 1)` on NVDA close prices.

2. **Sequence construction**:
   - Choose a **look_back = 60** days.
   - Build sequences \((X, y)\) where:
     - \(X_i\): last 60 scaled prices,
     - \(y_i\): next-day scaled price.

3. **Train/test split**:
   - 80% train, 20% test, no shuffle (to respect time ordering).

4. **Model**:
   - Sequential model:
     - LSTM(50 units, return_sequences=True),
     - LSTM(50 units, return_sequences=False),
     - Dense(1).
   - Loss: mean squared error, optimizer: Adam.
   - Trained for 50 epochs.

5. **Evaluation**:
   - Inverse-transform predictions back to price scale.
   - Compute **RMSE, MAE, R²** on the test set.

**Results (v1 LSTM):**

- **RMSE ≈ 3.33**
- **MAE ≈ 1.96**
- **R² ≈ 0.9964**

Visually, the predicted and actual curves almost overlap over the test period.

**Interpretation:**

- The LSTM clearly **outperforms both naïve and Prophet** on the chosen split.
- But such a high R² on noisy financial data suggests:
  - The model may be exploiting overlapping windows,
  - Data leakage is possible if evaluation is not strictly rolled forward,
  - Performance may not generalize to new regimes.

This motivated us in v2 to focus more on **evaluation protocols** (rolling forecasts) and **interpretable models**, rather than pushing deeper LSTM tuning.

---

## 3. Version 2 (Improved): Prophet-Centric with Better Preprocessing

v2 restructures everything around **Prophet** and careful **preprocessing**.

### 3.1 Baseline Prophet (Price Only)

First, we reproduce a baseline Prophet, now with a cleaner pipeline and a long history:

- Data: NVDA from 1999 onward.
- Split: earlier data for training, later for testing.
- Model: Prophet with default trend and seasonality on **raw price**.

**Result (v2 baseline):**

- **Baseline Prophet R² ≈ −1.22**

So baseline Prophet still performs worse than simply predicting the mean or naïve, confirming the difficulty of the problem.

### 3.2 Log-Transformed Prophet (Flexible Trend)

Next, we:

- Transform target: \(y = \log(\text{price})\).
- Use `changepoint_prior_scale` to allow a more flexible trend.

**Result (v2 log + flexible trend):**

- **R² ≈ −0.49**

This is a clear improvement over −1.22, but still negative, meaning the model is not yet competitive with simple baselines when evaluated as a single global fit.

### 3.3 Log + Seasonality

We then add **explicit weekly and yearly seasonality** on top of the log transform and flexible trend.

**Result (v2 enhanced Prophet):**

- **R² ≈ −0.49** (very close to the previous value)

So adding seasonality does **not** meaningfully improve performance on NVDA price, which is not strongly seasonal at the daily scale in the way business or demand series are.

### 3.4 Rolling Prophet (1-Day-Ahead, No Regressors)

To get a more realistic evaluation, we switch to a **rolling forecast**:

- Choose a split date (e.g., 2018–2023).
- Start with a training window up to the split.
- For each day in the test period:
  1. Fit Prophet on all data up to day \(t\).
  2. Forecast day \(t+1\).
  3. Append the true observation for day \(t+1\) to the training window.
  4. Repeat.

Using the **log-transformed price**, the rolling Prophet (no regressors) in v1 achieved:

- **Rolling Forecast R² ≈ 0.74**

This is a major improvement over the negative R² of global fits and shows that Prophet can be effective when used in an **online, refit-each-step** manner.

### 3.5 Rolling Prophet with Regressors (SPY, Volume, VIX)

In v2 we further extend the rolling setup to include regressors:

- `add_regressor("SPY")`
- `add_regressor("Volume")`
- `add_regressor("VIX")`

All are log-transformed and aligned with NVDA.

In v1, a “Rolling Forecast with Regressors” (SPY + Volume) produced:

- **Rolling Forecast with Regressors R² ≈ 0.55**

In v2 we also add VIX for a three-regressor setup (SPY + Volume + VIX). The code computes R² for each model, but the last run’s numbers aren’t stored in the notebook outputs. Conceptually:

- SPY and VIX provide **macro-market context**,
- Volume encodes **liquidity / attention**,
- The rolling refit ensures that the model uses only past information at each step.

Even though the specific R² for SPY + Volume + VIX isn’t captured in the saved outputs, the design is consistent with financial forecasting practice (multi-factor models).

### 3.6 Naïve Baseline in v2

v2 also implements a **naïve baseline on the same test window** as the Prophet models:

- First test-day prediction = last training-day value,
- After that, predict the previous day’s observed price for each test day.

This provides a fair, same-period comparison between:

- Naïve,
- Global Prophet variants,
- Rolling Prophet variants.

---

## 4. Comparison of v1 vs v2: What Changed and Why

### 4.1 Modeling Scope

- **v1**:
  - Broad exploration.
  - Models: naïve, Prophet (with holidays and AI event windows), LSTM.
  - Focus: show strong LSTM performance vs Prophet and naïve.

- **v2**:
  - Focused, Prophet-centric pipeline.
  - Models: multiple Prophet variants + naïve.
  - Focus: better **preprocessing**, **feature engineering**, and **evaluation protocol** (rolling).

### 4.2 Key Quantitative Takeaways

- Naïve baseline is **very strong** (R² ≈ 0.95 for 1-day ahead in v1).
- Prophet, if used as a single global fit on raw price, is **poor** (R² ≈ −1 to −76 depending on setup).
- LSTM in v1 achieves **R² ≈ 0.996**, but with caveats about generalization.
- Rolling Prophet in v1/v2 shows that **evaluation protocol matters**:
  - Rolling no-regressor Prophet: R² ≈ 0.74
  - Rolling with regressors: R² ≈ 0.55 on one configuration.

### 4.3 Why We Changed the Code in v2

We made v2 to:

1. **Clean up the pipeline**:
   - Shared helper functions (`build_prophet_df`, etc.).
   - Centralized data loading and preprocessing.
   - Consistent train/test splits.

2. **Emphasize interpretability**:
   - Prophet fits with trend / seasonality / regressor plots.
   - Easier to explain in an EE344 report than a deep LSTM black box.

3. **Improve evaluation realism**:
   - Rolling forecasts instead of a single static split.
   - Avoid overly optimistic metrics.

4. **Highlight preprocessing decisions**:
   - Log transforms,
   - Extra regressors (SPY, Volume, VIX),
   - Forward-filling and date alignment.

v1 shows “how far” a deep model can push accuracy on one split; v2 shows a more **robust, interpretable ML pipeline** that aligns with course goals and can be maintained or extended by others.

---

## 5. Lessons Learned

1. **Baselines first**: For financial data, naïve forecasts set a very high bar.
2. **Prophet needs help**: On raw NVDA prices, Prophet alone performs poorly; log transforms, exogenous regressors, and rolling refits substantially improve things.
3. **Deep models can overfit**: LSTMs can produce impressive R² but must be evaluated with caution.
4. **Evaluation protocol is part of the method**: Switching from a single train/test split to rolling evaluation changed our understanding of Prophet’s performance.
5. **Preprocessing is a model choice**: Log scaling, merging regressors, and how we handle missing days have a direct impact on model behavior and fairness of comparisons.

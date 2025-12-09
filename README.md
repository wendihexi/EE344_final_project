# EE344 Final Project – NVDA Stock Forecasting (Prophet & LSTM)

This repository contains our EE344 final project on **time-series forecasting for NVIDIA (NVDA) daily stock prices** using:

- Simple **naïve baselines**,
- **Prophet** (additive forecasting model),
- A deep **LSTM** model (in the original version).

We have **two main notebook versions**:

- **v1: `Prophet_testing.ipynb`**  
  Original final-project notebook with naïve vs Prophet vs LSTM.

- **v2: `Prophet_testing_v2.ipynb`**  
  Improved, Prophet-focused notebook with better preprocessing and rolling forecasts.

---

## Repository Contents

- `Prophet_testing.ipynb`  
  Original final notebook:
  - Loads NVDA data from Yahoo Finance.
  - Builds a naïve persistence baseline.
  - Trains several Prophet models with US holidays and custom “NVDA AI event” holidays.
  - Trains an LSTM model on scaled NVDA prices.
  - Reports MAE/RMSE/R² metrics and plots for each model.

- `Prophet_testing_v2.ipynb`  
  Improved notebook:
  - Centralizes **data loading and preprocessing** (NVDA, SPY, VIX, Volume).
  - Implements:
    - Baseline Prophet on price,
    - Log-transformed Prophet with flexible trend,
    - Enhanced Prophet with explicit seasonality,
    - Rolling Prophet (1-day-ahead) without regressors,
    - Rolling Prophet with exogenous regressors (SPY, Volume, VIX),
    - Naïve baseline on the same test period.
  - Compares models using R² and forecast plots on the test period.

-  `literature_review.md`  
  Background on forecasting methods (naïve, Prophet, LSTM) and how the literature informed our design.

-  `methods_and_analysis.md`  
  Detailed description of preprocessing, methods, and a comparison of v1 vs v2.

---

## Dependencies

Both notebooks assume Python 3.10+ and use the following main packages:

- `numpy`
- `pandas`
- `matplotlib`
- `yfinance`
- `prophet`
- `scikit-learn`
- `tqdm` (for progress bars, mainly in v2)
- `tensorflow` (for the LSTM in v1)

### Example `requirements.txt` (pip)

```text
numpy
pandas
matplotlib
yfinance
prophet
scikit-learn
tqdm
tensorflow

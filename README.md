# Short-Term Electricity Consumption Forecasting

An end-to-end time-series forecasting project for predicting short-term electricity demand using historical consumption patterns, weather conditions, temporal features, lag variables, rolling statistics, XGBoost, and LSTM.

## Problem Statement

Accurate short-term electricity demand forecasting is important for efficient power generation, grid management, and resource planning.

The objective of this project is to:

- Analyze electricity consumption patterns over time
- Identify daily, weekly, and seasonal demand patterns
- Examine the relationship between weather conditions and electricity consumption
- Test the stationarity and temporal dependence of the series
- Engineer time-series features such as lags and rolling statistics
- Develop machine learning and deep learning forecasting models
- Compare XGBoost and LSTM performance
- Identify the most important predictors of short-term electricity demand

## Dataset

The dataset contains **52,416 observations recorded at 10-minute intervals** across 2017.

It contains:

### Weather Features
- Temperature
- Humidity
- Wind Speed
- General Diffuse Flows
- Diffuse Flows

### Electricity Consumption
- Zone 1 Consumption
- Zone 2 Consumption
- Zone 3 Consumption

### Target Variable

`Total_Consumption`

The target was created by aggregating electricity consumption across the three distribution zones:

`Total_Consumption = Zone1 + Zone2 + Zone3`

This provides a single measure of overall electricity demand. :contentReference[oaicite:1]{index=1}

## Data Quality & Preparation

The dataset was checked for:

- Missing values
- Duplicate observations
- Duplicate timestamps
- Irregular time intervals
- Incorrect data types

No duplicate rows or duplicate timestamps were found. The data spans from **1 January 2017 to 30 December 2017**, with observations consistently recorded every **10 minutes**. :contentReference[oaicite:2]{index=2}

The `Datetime` variable was converted to datetime format, sorted chronologically, and set as the time-series index.

## Feature Engineering

Calendar-based features were extracted from the timestamp:

- Hour
- Day
- Month
- Weekday
- Weekend
- Quarter

These features capture daily, weekly, monthly, and quarterly patterns in electricity demand. :contentReference[oaicite:3]{index=3}

### Time-Series Features

Additional forecasting features were engineered from historical consumption:

- Lag 1
- Lag 6
- Lag 18
- Lag 144
- Rolling Mean 6
- Rolling Mean 144
- Rolling Standard Deviation 6

A lag of **144 observations represents one full day**, because the dataset contains six observations per hour and 144 observations per 24 hours.

## Exploratory Data Analysis

The project investigated electricity demand from both statistical and time-series perspectives.

### Key Findings

**Hourly Pattern**

Electricity demand decreases during the night and increases during the day, reaching its highest average level around **20:00**. :contentReference[oaicite:4]{index=4}

**Seasonality**

Demand increased during the middle of the year, reached high levels during July and August, and declined toward December. :contentReference[oaicite:5]{index=5}

**Weekday vs Weekend**

Average weekday consumption was approximately **71,999 units**, compared with **69,282 units** on weekends. :contentReference[oaicite:6]{index=6}

**Temperature**

Temperature showed the strongest weather-related relationship with total consumption:

`Correlation = 0.4882`

Humidity showed a negative correlation of `-0.2991`. :contentReference[oaicite:7]{index=7}

## Outlier Analysis

The target variable showed moderate positive skewness:

- Skewness: **0.519**
- Kurtosis: **-0.233**

The IQR method identified **150 observations (0.29%)** as potential outliers.

These observations were **retained** because they may represent genuine peak electricity-demand periods rather than erroneous measurements. :contentReference[oaicite:8]{index=8}

## Time-Series Analysis

### Rolling Statistics

A 24-hour rolling window of **144 observations** was used to examine changes in the mean and variability of electricity consumption.

The rolling analysis showed that both the level and variability of demand changed across the year, suggesting non-stationary behavior. :contentReference[oaicite:9]{index=9}

### ADF Test

The Augmented Dickey-Fuller test produced:

- ADF Statistic: **-26.5676**
- p-value: **0.0000**

The ADF test rejected the null hypothesis of a unit root. :contentReference[oaicite:10]{index=10}

### KPSS Test

The KPSS test produced:

- KPSS Statistic: **5.7828**
- p-value: **< 0.01**

The KPSS test rejected the null hypothesis of stationarity.

Because ADF and KPSS gave conflicting conclusions, additional seasonal and autocorrelation analysis was performed rather than relying on a single test. :contentReference[oaicite:11]{index=11}

## Seasonal Decomposition

An additive seasonal decomposition with a period of **144 observations** was performed.

The decomposition identified:

- Trend
- Daily seasonal component
- Residual component

A strong and regular daily seasonal pattern was observed. :contentReference[oaicite:12]{index=12}

## ACF & PACF Analysis

The ACF analysis showed:

- Strong autocorrelation at short lags
- Pronounced seasonal dependence around lag 144
- Strong daily periodicity

The PACF showed a strong spike at lag 1 along with seasonal dependence around lag 144.

These findings confirmed the importance of historical consumption and daily seasonality in forecasting. :contentReference[oaicite:13]{index=13}

# Machine Learning Forecasting

## XGBoost

An XGBoost regression model was developed using:

- Weather variables
- Calendar features
- Lag features
- Rolling statistics

The model was evaluated on a chronological test set to preserve the time-series structure.

### XGBoost Performance

| Metric | Test Performance |
|---|---:|
| MAE | **415.55** |
| RMSE | **600.34** |
| MAPE | **0.66%** |
| R² | **0.9982** |

The XGBoost model achieved excellent short-term forecasting performance. :contentReference[oaicite:14]{index=14}

## Feature Importance

The most influential features were:

| Feature | Importance |
|---|---:|
| Lag_1 | **0.4519** |
| Lag_144 | **0.3616** |
| Rolling_Mean_6 | **0.1709** |

These results demonstrate that recent consumption, previous-day consumption, and short-term rolling demand are the strongest predictors of the next electricity-consumption value. :contentReference[oaicite:15]{index=15}

### Interpretation

The importance of `Lag_1` shows that electricity demand is strongly dependent on the immediately preceding 10-minute observation.

The importance of `Lag_144` demonstrates the presence of a strong daily seasonal pattern.

`Rolling_Mean_6` captures short-term local demand trends.

## Univariate LSTM

A univariate LSTM model was developed using only historical `Total_Consumption`.

The data was divided chronologically into:

- Training: **41,817 observations**
- Testing: **10,455 observations**

This approach allowed the LSTM to learn temporal dependencies without using external weather variables. :contentReference[oaicite:16]{index=16}

### LSTM Performance

| Metric | LSTM |
|---|---:|
| MAE | 529.10 |
| RMSE | 808.95 |
| MAPE | 0.83% |
| R² | 0.9967 |

The LSTM successfully learned the temporal structure of electricity consumption but performed slightly below XGBoost. :contentReference[oaicite:17]{index=17}

# Model Comparison

| Model | MAE | RMSE | MAPE | R² |
|---|---:|---:|---:|---:|
| **XGBoost** | **415.55** | **600.34** | **0.66%** | **0.9982** |
| LSTM | 529.10 | 808.95 | 0.83% | 0.9967 |

### Final Model

**XGBoost was selected as the final forecasting model** because it achieved lower MAE, lower RMSE, lower MAPE, and higher R² than the univariate LSTM. :contentReference[oaicite:18]{index=18}

# Business Insights

### 1. Short-Term Demand Forecasting

The model can provide short-term electricity demand estimates to support operational planning.

### 2. Daily Load Planning

The strong daily seasonality indicates that electricity demand can be anticipated using previous-day consumption patterns.

### 3. Peak Demand Management

The hourly and seasonal analyses identify periods of elevated electricity consumption, supporting peak-load planning.

### 4. Weather-Aware Forecasting

Temperature and humidity provide additional information for electricity-demand prediction.

### 5. Data-Driven Grid Planning

Accurate short-term forecasts can support resource allocation and electricity supply planning.

# Project Workflow

```text
Data Loading
      ↓
Data Quality Checks
      ↓
Datetime Processing
      ↓
Target Creation
      ↓
Calendar Feature Engineering
      ↓
Exploratory Data Analysis
      ↓
Time-Series Analysis
      ↓
ADF & KPSS Tests
      ↓
Seasonal Decomposition
      ↓
ACF & PACF Analysis
      ↓
Lag & Rolling Features
      ↓
Chronological Train-Test Split
      ↓
XGBoost Forecasting
      ↓
Univariate LSTM Forecasting
      ↓
Model Evaluation
      ↓
Model Comparison
      ↓
Feature Importance
      ↓
Final XGBoost Model

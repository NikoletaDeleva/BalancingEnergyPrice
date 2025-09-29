# Baltic Balancing Energy Price Forecasting

This project builds a forecasting model for Baltic balancing energy prices using hourly time-series data from multiple sources: the Baltic Transparency Dashboard, ENTSO-E day-ahead electricity prices, and historical weather data from Open-Meteo. The aim is to predict the next 24 hours of balancing energy prices each day, and evaluate the performance on September 2024 data.

---

## Goal

* **Target**: Balancing energy price (treated as a single time series)
* **Frequency**: Hourly
* **Forecast Objective**: Daily prediction of the next 24 hours
* **Evaluation Window**: September 1 – September 30, 2024

---

## Data Sources & Preparation

### 1. **Balancing Energy Prices (Baltic Transparency Dashboard)**

* Downloaded via public API as CSV
* Parses and unifies upward/downward prices
* Cleans European decimal formats and normalizes timestamps
* Aggregates values across all directions and Baltic countries using the **median per hour**
* Interpolates missing values with limited fill for continuity

**Target generation:**

* Converts raw prices to a single time-series of hourly values (`y`)
* Creates multi-horizon supervised targets (`target_0` to `target_23`) representing prices from 0 to 23 hours into the future
* Extracts origin-time features using lags and rolling statistics
* Adds calendar-based features (hour, weekday, weekend flag)

### 2. **ENTSO-E Day-Ahead Electricity Prices**

* Fetched using the official ENTSO-E API (requires token)
* Downloads data per country and time window
* Standardizes to hourly granularity and local time zone
* Aggregates Baltic countries using the mean per timestamp
* Interpolates missing values

**Features extracted:**

* Lag features (e.g., lag_0 = today's DA price at forecast time)
* Rolling mean and std (24h, 168h)
* Price differences (e.g., last hour, last day)

### 3. **Weather Data (Open-Meteo Historical API)**

* Fetches hourly weather variables for geographic coordinates
* Aggregates multiple locations per country (with optional weights)
* Normalizes time to local time zone
* Interpolates and fills missing weather data

**Weather features include:**

* Lags (1h, 24h, 168h)
* Rolling mean and std (24h, 168h)
* Variables used: `temperature_2m`, `windspeed_10m`, `shortwave_radiation`

---

## Modeling Approach

We implemented **two modeling strategies** using XGBoost regressors:

### 1. `SingleModelForecaster`

* Trains a **single multi-output model** to jointly predict all 24 hourly horizons
* Feature matrix includes engineered features and static context per origin
* Hyperparameter tuning via `RandomizedSearchCV` with time-series split
* Allows shared representation learning across all horizons

### 2. `MultiModelForecaster`

* Trains **one independent XGBoost model per forecast horizon** (e.g., one for 1h ahead, one for 2h ahead, ..., 24h ahead)
* Each model is trained and tuned individually
* Allows horizon-specific parameter optimization (e.g., short-term vs. long-term behavior)
* Tuning grid based on earlier experiments and best practices for time-series forecasting

---

## Evaluation

### Metrics

We evaluate both training and testing performance using:

* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Squared Error)**

Each metric is computed per horizon (t0–t23).

### Visualizations

* Heatmaps of MAE and RMSE across all 24 horizons

---

##  Future Work / Optimizations and Improvments

### Engineering / Automation
* **CLI** tool or **YAML**-based pipeline for fully automated runs from config
* **Docker** container to encapsulate data fetching, preprocessing, training, and evaluation
* **MLFlow** or **DVC** integration for experiment tracking, model registry, and versioning
* Pre-scheduled retraining pipeline (e.g., daily at midnight) using **Airflow**

### Data Improvements
* **Real-time** integration with Baltic dashboard, ENTSO-E, and Open-Meteo APIs
* Broaden date range to capture seasonal effects (e.g., train from 2023–2024)
* Geospatial interpolation for weather: use actual location of substations

### Deployment Considerations
* **REST API** for getting forecasts programmatically
* **Grafana** dashboard for visualization
* **Fallback** logic if weather or DA prices are missing

### Feature Engineering
* Holiday / weekend flags, especially for the Baltics
* Lagged weather impact cross features (e.g., lagged windspeed × solar radiation)
* Demand proxy data: build from ENTSO-E load or historical averages
* Wind & solar capacity forecast from external sources

### Modeling Enhancements
* Ensemble approaches: e.g., average/mix between single-model and multi-model forecasts
* Switch from RandomizedSearchCV to Optuna or Hyperopt
* Custom loss functions: e.g., penalize overprediction more than underprediction
* Target transformation (log, power) to stabilize variance or improve RMSE
* Error analysis by hour: identify hard-to-predict time slots (e.g., night hours)

### Evaluation & Explainability
* Hourly vs. daily aggregated metrics (MAE, MAPE, RMSE) comparison
* SHAP values or feature importance plots per hour
* Forecast vs. actual timeline plots for each day in September
* Highlight anomaly days (e.g., extreme prices or weather) with model confidence


---

## Dependencies

* `pandas`, `numpy`, `xgboost`, `scikit-learn`, `matplotlib`, `seaborn`, `requests`


----

## Usage

* Fill the parameters in the params.py 
* Execute from the executable folder first dataPrep.py then forecast.py

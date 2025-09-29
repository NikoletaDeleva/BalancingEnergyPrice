import sys

sys.path.append("BalancingEnergyPrice")

import pandas as pd
from pathlib import Path

# Import custom modules
import params as p
from modules.utils import time_split
from forecaster.singleModelForecatser import SingleModelForecaster
from forecaster.multiModelForecaster import MultiModelForecaster

# ----------------- LOAD DATA -----------------
# Load pre-prepared daily data
X_daily = pd.read_csv(Path(p.output_dir) / "X_daily.csv", index_col=0, parse_dates=True)
Y_daily = pd.read_csv(Path(p.output_dir) / "Y_daily.csv", index_col=0, parse_dates=True)

# ----------------- SPLIT train vs test -----------------
# Given test start and end dates, split the DataFrame into 1-year train and 1-month test.
X_train, X_test = time_split(X_daily, start=p.START, end=p.END, tz=p.TZ)
Y_train, Y_test = time_split(Y_daily, start=p.START, end=p.END, tz=p.TZ)

# ----------------- TUNE + FIT + FORECAST -----------------
# Common parameter grid for tuning
raw_param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [3, 4, 5],
    "min_child_weight": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0, 0.1, 0.5, 1],
    "reg_lambda": [1, 1.5, 2, 3],
}

wrapped_param_grid = {f"estimator__{k}": v for k, v in raw_param_grid.items()}

# ---------------- SINGLE-MODEL FORECASTER -----------------

mo = SingleModelForecaster()

# Tune
best_params = mo.tune(X_train, Y_train, param_grid=wrapped_param_grid, n_iter=10)
print("Best params:", best_params)

# Final fit & predict
mo.fit(X_train, Y_train)
Y_pred = mo.predict(X_test)

# Evaluate
mo.evaluate(X_train, Y_train)

mo.evaluate(X_test, Y_test)

# # ---------------- MULTI-MODEL FORECASTER -----------------

mm = MultiModelForecaster()

# Tune
best_params = mm.tune(X_train, Y_train, param_grid=raw_param_grid, n_iter=10)
print("Best params:", best_params)

mm.fit(X_train, Y_train)

# Predict
Y_pred = mm.predict(X_test)

# Evaluate
mm.evaluate(X_train, Y_train)

mm.evaluate(X_test, Y_test)

# Save feature importances
mm.save_feature_importance(Path(p.output_dir))

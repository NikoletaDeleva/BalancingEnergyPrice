from typing import Optional, Tuple, Union
import pandas as pd
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error


class SingleModelForecaster:
    """
    Forecast 24-hour targets using a single multi-output XGBoost model.
    """

    def __init__(self, xgb_params: Optional[dict] = None, random_state: int = 42):
        self.xgb_params = xgb_params or {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_state,
        }
        self.model = MultiOutputRegressor(XGBRegressor(**self.xgb_params))

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> None:
        """Fit the multi-output model."""
        self.model.fit(X, Y)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        preds = self.model.predict(X)
        return pd.DataFrame(
            preds, index=X.index, columns=[f"yhat_t{h}" for h in range(preds.shape[1])]
        )

    def evaluate(
        self,
        X: pd.DataFrame,
        Y_true: pd.DataFrame,
        metrics: Optional[Union[str, list]] = None,
    ) -> pd.DataFrame:
        if isinstance(metrics, str):
            metrics = [metrics]
        metrics = metrics or ["mae", "rmse"]

        Y_pred = self.predict(X)
        scores = {}

        for metric in metrics:
            if metric == "mae":
                scores["mae"] = [
                    mean_absolute_error(Y_true.iloc[:, h], Y_pred.iloc[:, h])
                    for h in range(Y_pred.shape[1])
                ]
            elif metric == "rmse":
                scores["rmse"] = [
                    mean_squared_error(Y_true.iloc[:, h], Y_pred.iloc[:, h])
                    for h in range(Y_pred.shape[1])
                ]
            else:
                raise ValueError(f"Unsupported metric: {metric}")

        return pd.DataFrame(scores, index=[f"t{h}" for h in range(Y_pred.shape[1])])

    def tune(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        param_grid: dict,
        n_iter: int = 20,
        cv: int = 3,
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: int = 1,
    ) -> dict:
        """
        Tune hyperparameters using RandomizedSearchCV with TimeSeriesSplit.
        """
        base_model = MultiOutputRegressor(XGBRegressor(random_state=random_state))
        tscv = TimeSeriesSplit(n_splits=cv)

        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=tscv,
            scoring="neg_mean_absolute_error",
            verbose=verbose,
            n_jobs=n_jobs,
            random_state=random_state,
        )

        search.fit(X, Y)
        self.model = search.best_estimator_
        return search.best_params_

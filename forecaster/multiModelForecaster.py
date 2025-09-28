from fileinput import filename
import os
from typing import Optional, Union, Iterable
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


class MultiModelForecaster:
    """
    Trains 24 separate XGBoost models, one per hour (horizon).
    Each model predicts y[t+h] from a shared feature set X[t].

    Supports fit, predict, evaluate, and tune.
    """

    def __init__(
        self,
        horizons: Iterable[int] = range(24),
        xgb_params: Optional[dict] = None,
        random_state: int = 42,
    ):
        self.horizons = list(horizons)
        self.xgb_params = xgb_params or {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_state,
        }
        self.models = {}
        self.best_params_by_horizon = {}

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> None:
        for h in self.horizons:
            best_params = self.best_params_by_horizon.get(h, self.xgb_params)
            model = XGBRegressor(**best_params)
            model.fit(X, Y.iloc[:, h])
            self.models[h] = model

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        preds = {}
        for h in self.horizons:
            preds[f"yhat_t{h}"] = self.models[h].predict(X)
        return pd.DataFrame(preds, index=X.index)

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
                    mean_absolute_error(Y_true.iloc[:, h], Y_pred[f"yhat_t{h}"])
                    for h in self.horizons
                ]
            elif metric == "rmse":
                scores["rmse"] = [
                    mean_squared_error(Y_true.iloc[:, h], Y_pred[f"yhat_t{h}"])
                    for h in self.horizons
                ]
            else:
                raise ValueError(f"Unsupported metric: {metric}")

        return pd.DataFrame(scores, index=[f"t{h}" for h in self.horizons])

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
        Tune each hourly model separately.
        Returns a dictionary of best parameters per horizon.
        """
        best_params = {}
        tscv = TimeSeriesSplit(n_splits=cv)

        for h in self.horizons:
            base_model = XGBRegressor(random_state=random_state)
            search = RandomizedSearchCV(
                base_model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=tscv,
                scoring="neg_mean_absolute_error",
                n_jobs=n_jobs,
                verbose=verbose,
                random_state=random_state,
            )
            search.fit(X, Y.iloc[:, h])
            self.models[h] = search.best_estimator_
            best_params[h] = search.best_params_
            self.best_params_by_horizon[h] = search.best_params_

        return best_params

    def save_feature_importance(self, output_path: str, normalize: bool = True) -> None:
        """
        Save feature importances per horizon to a CSV file.

        Args:
            filename: Output CSV file path.
            normalize: Whether to normalize importances across each horizon.
        """
        # Get features from one model
        all_features = list(self.models.values())[0].feature_names_in_
        importance_df = pd.DataFrame(index=all_features)

        # Collect importances for each horizon
        for h, model in self.models.items():
            importance_df[f"t{h}"] = model.feature_importances_

        if normalize:
            importance_df = importance_df.div(importance_df.max(axis=0), axis=1)

        # Save to CSV
        importance_df.to_csv(os.path.join(output_path, "feature_importance.csv"))
        print(
            f"Feature importances saved to {os.path.join(output_path, 'feature_importance.csv')}"
        )

        # Plot heatmap
        plt.figure(figsize=(14, max(6, len(all_features) // 2)))
        sns.heatmap(importance_df, cmap="viridis", linewidths=0.5)
        plt.title("Feature Importance Heatmap (per Horizon)")
        plt.xlabel("Horizon")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "feature_importance.png"), dpi=300)
        plt.close()

        print(f"Heatmap saved to {os.path.join(output_path, 'feature_importance.png')}")

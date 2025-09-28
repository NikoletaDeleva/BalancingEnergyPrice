import time
import requests
import pandas as pd
from typing import Dict, List, Iterable, Optional


class MateoData:
    OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

    ALLOWED_ARCHIVE_VARS = {
        "temperature_2m",
        "relative_humidity_2m",
        "dewpoint_2m",
        "apparent_temperature",
        "precipitation",
        "rain",
        "snowfall",
        "snow_depth",
        "windspeed_10m",
        "winddirection_10m",
        "shortwave_radiation",  # prefer this (not surface_solar_radiation)
        "cloudcover",
        "surface_pressure",
        "pressure_msl",
    }

    def __init__(
        self,
        start_date: str,
        end_date: str,
        tz: str,
        area_points: Dict[str, List[tuple]],
        hourly_vars: Iterable[str] = (
            "temperature_2m",
            "windspeed_10m",
            "shortwave_radiation",
        ),
        weights: Optional[Dict[str, List[float]]] = None,
        agg: str = "mean",
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.tz = tz
        self.area_points = area_points
        self.hourly_vars = list(hourly_vars)
        self.weights = weights or {}
        self.agg = agg

        # Fail fast if bad variables were passed
        self._validate_vars(self.hourly_vars)

        # Validate weights lengths if provided
        for area, pts in self.area_points.items():
            if area in self.weights:
                w = self.weights[area]
                if len(w) != len(pts):
                    raise ValueError(
                        f"Weights for area '{area}' have length {len(w)} but there are {len(pts)} points."
                    )
                if sum(w) == 0:
                    raise ValueError(f"Weights for area '{area}' sum to zero.")

    # ----------------------- Validation helpers -----------------------

    def _validate_vars(self, hourly: Iterable[str]):
        bad = [v for v in hourly if v not in self.ALLOWED_ARCHIVE_VARS]
        if bad:
            raise ValueError(
                f"Unsupported archive variables: {bad}. Allowed: {sorted(self.ALLOWED_ARCHIVE_VARS)}"
            )

    @staticmethod
    def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Expected df.index to be a DatetimeIndex")
        return df

    # ----------------------- Data fetch -----------------------

    def fetch_openmeteo_hourly(
        self,
        latitude: float,
        longitude: float,
        *,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        timeout: int = 60,
    ) -> pd.DataFrame:
        """
        Fetch hourly historical weather (archive) for one lat/lon.
        Request UTC and convert to target tz to avoid DST duplicates.
        Includes simple retry/backoff for transient errors.
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": self.start_date[:10],  # Open-Meteo expects YYYY-MM-DD
            "end_date": self.end_date[:10],  # inclusive
            "hourly": ",".join(self.hourly_vars),
            "timezone": "UTC",  # request UTC to avoid DST duplicates
            "timeformat": "iso8601",
        }

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                r = requests.get(
                    self.OPEN_METEO_ARCHIVE, params=params, timeout=timeout
                )
                # Retry on 5xx; raise for others
                if 500 <= r.status_code < 600:
                    raise requests.HTTPError(
                        f"Open-Meteo {r.status_code}: {r.text[:500]}", response=r
                    )
                if r.status_code >= 400:
                    # Non-retryable 4xx—surface the message
                    try:
                        msg = r.json()
                    except Exception:
                        msg = r.text[:500]
                    raise requests.HTTPError(
                        f"Open-Meteo {r.status_code}: {msg}", response=r
                    )
                data = r.json()
                break
            except (
                requests.Timeout,
                requests.ConnectionError,
                requests.HTTPError,
            ) as e:
                last_err = e
                if attempt == max_retries:
                    raise
                time.sleep(backoff_base * (2 ** (attempt - 1)))
        else:
            # Shouldn't reach here, but keep mypy happy
            raise last_err or RuntimeError("Unknown error fetching Open-Meteo data.")

        if "hourly" not in data or "time" not in data["hourly"]:
            raise ValueError(
                f"Open-Meteo: missing 'hourly' data. Head: {str(data)[:400]}"
            )

        df = pd.DataFrame(data["hourly"])

        # Parse UTC → convert to desired tz (keeps uniqueness across DST)
        ts = pd.to_datetime(df["time"], utc=True).dt.tz_convert(self.tz)
        df = df.drop(columns=["time"])
        df.insert(0, "timestamp", ts)

        # Ensure numeric columns are numeric
        for c in df.columns:
            if c != "timestamp":
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Collapse any accidental duplicate timestamps
        df = df.groupby("timestamp", as_index=False).mean(numeric_only=True)

        # Hourly frequency
        df = df.set_index("timestamp").sort_index().asfreq("h")
        return df

    def build_area_weather_multi(self) -> pd.DataFrame:
        """
        Fetch and aggregate hourly weather per area.
        - Supports optional per-area weights (must match number of points).
        - Aggregates by mean/median or weighted mean.
        """
        out = []
        for area, pts in self.area_points.items():
            per_point = []
            for i, (lat, lon) in enumerate(pts):
                dfp = self.fetch_openmeteo_hourly(lat, lon).copy()
                dfp["point_id"] = i
                per_point.append(dfp)

            area_df = pd.concat(
                per_point
            ).reset_index()  # columns: timestamp, <vars>, point_id

            var_cols = [
                c for c in area_df.columns if c not in ("timestamp", "point_id")
            ]

            if area in self.weights:
                w_raw = pd.Series(
                    self.weights[area], index=range(len(pts)), dtype=float
                )
                w = w_raw / w_raw.sum()

                def _wavg(g: pd.DataFrame) -> pd.Series:
                    wm = g["point_id"].map(w)
                    return pd.Series(
                        {c: (g[c] * wm).sum(skipna=True) for c in var_cols}
                    )

                agg_df = (
                    area_df.groupby("timestamp", as_index=False)
                    .apply(_wavg)
                    .reset_index(drop=True)
                )

            elif self.agg == "median":
                agg_df = (
                    area_df.groupby("timestamp", as_index=False)[var_cols]
                    .median()
                    .reset_index(drop=True)
                )
            else:
                agg_df = (
                    area_df.groupby("timestamp", as_index=False)[var_cols]
                    .mean()
                    .reset_index(drop=True)
                )

            agg_df["area"] = area
            out.append(agg_df)

        return pd.concat(out, ignore_index=True).sort_values(["area", "timestamp"])

    def make_weather_hourly(
        self, wx_df: pd.DataFrame, *, fill_limit: int = 3
    ) -> pd.DataFrame:
        """
        Convert multi-area weather into single hourly frame: mean across areas, interpolated.
        """
        if "timestamp" not in wx_df.columns:
            raise ValueError("wx_df must contain 'timestamp'")
        wx = wx_df.copy().set_index("timestamp").sort_index()

        # Aggregate across areas → mean per hour
        wx_agg = (
            wx.groupby(wx.index)[list(self.hourly_vars)]
            .mean(numeric_only=True)
            .sort_index()
        )

        # Full hourly grid over [start_date, end_date)
        start_ts = pd.Timestamp(f"{self.start_date} 00:00", tz=self.tz)
        end_ex = pd.Timestamp(f"{self.end_date} 00:00", tz=self.tz) + pd.Timedelta(
            days=1
        )
        idx = pd.date_range(start_ts, end_ex, freq="H", inclusive="left")

        return (
            wx_agg.reindex(idx)
            .interpolate(limit=fill_limit, limit_direction="both")
            .sort_index()
        )

    # ----------------------- Feature engineering -----------------------
    def make_weather_features(
        self,
        wx_agg: pd.DataFrame,
        *,
        lags=(1, 24, 168),
        rolls=(24, 168),
        prefix="fe_wx_",
    ) -> pd.DataFrame:
        """
        Input: hourly weather DataFrame (already aggregated + interpolated).
        Output: engineered lag, rolling, and std features.
        """
        wx_agg = wx_agg.sort_index()
        out = {}

        for v in self.hourly_vars:
            s = wx_agg[v]
            # Lags
            for L in lags:
                out[f"{prefix}{v}_lag_{L}"] = s.shift(L)
            # Rolling
            for W in rolls:
                out[f"{prefix}{v}_roll_mean_{W}"] = s.rolling(W).mean().shift(1)
                out[f"{prefix}{v}_roll_std_{W}"] = s.rolling(W).std(ddof=0).shift(1)

        return pd.DataFrame(out, index=wx_agg.index).sort_index()

import requests
import pandas as pd
import csv
from pathlib import Path
from urllib.parse import urlencode
from typing import Iterable


# ----------------------------- Baltic dashboard CSV -----------------------------
class BalticData:

    API_BASE = "https://api-baltic.transparency-dashboard.eu/api/v1/export"
    DATASET_ID = "balancing_energy_prices"

    def __init__(
        self,
        start_date: str,
        end_date: str,
        tz_api: str,
        tz_market: str,
        out_dir: Path,
        out_name: str,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.tz_api = tz_api
        self.tz_market = tz_market
        self.out_dir = out_dir
        self.out_name = out_name

    def build_url(self, output_format: str = "csv") -> str:
        """
        Dates must be ISO-like: YYYY-MM-DDTHH:MM (e.g., 2024-09-01T00:00).
        tz example: 'EET' (as used by the dashboard).
        """
        # Build local ISO strings for BTD (end exclusive)
        if len(self.start_date) == 10 and len(self.end_date) == 10:
            start_iso = f"{self.start_date}T00:00"
            end_dt = pd.Timestamp(
                f"{self.end_date} 00:00", tz=self.tz_market
            ) + pd.Timedelta(days=1)
            end_iso = end_dt.strftime("%Y-%m-%dT%H:%M")
        else:
            start_iso = self.start_date
            end_iso = self.end_date

        params = {
            "id": self.DATASET_ID,
            "start_date": start_iso,
            "end_date": end_iso,
            "output_time_zone": self.tz_api,
            "output_format": output_format,
        }
        return f"{self.API_BASE}?{urlencode(params)}"

    @staticmethod
    def download_file(url: str, out_path: Path, chunk: int = 8192) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        headers = {"User-Agent": "Mozilla/5.0"}
        with requests.get(url, headers=headers, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for part in r.iter_content(chunk_size=chunk):
                    if part:
                        f.write(part)
        return out_path

    @staticmethod
    def sniff_delimiter(sample_text: str) -> str:
        try:
            dialect = csv.Sniffer().sniff(sample_text, delimiters=[",", ";", "|", "\t"])
            return dialect.delimiter
        except Exception:
            return ";"  # common in EU datasets

    def read_csv_auto(self, path: Path) -> pd.DataFrame:
        with open(path, "rb") as f:
            head = f.read(65536).decode("utf-8-sig", errors="replace")
        sep = self.sniff_delimiter(head)
        return pd.read_csv(path, sep=sep, encoding="utf-8-sig")

    def fetch_balancing_prices(self) -> pd.DataFrame:
        """
        If you pass date-only (YYYY-MM-DD), we will take whole days:
        start = 00:00 local that day
        end   = 00:00 local of the *next* day (exclusive)
        If you pass full 'YYYY-MM-DDTHH:MM', we use as-is.
        """
        url = self.build_url()
        out_path = Path(self.out_dir / self.out_name)
        self.download_file(url, out_path)
        df = self.read_csv_auto(out_path)
        return df

    @staticmethod
    def clean_balancing_df(df_raw: pd.DataFrame, tz: str) -> pd.DataFrame:
        # Parse datetimes like "YYYY-MM-DD H:MM"
        for c in ["datetime_from", "datetime_to"]:
            df_raw[c] = pd.to_datetime(
                df_raw[c], format="%Y-%m-%d %H:%M", errors="coerce"
            )

        df = df_raw.rename(columns={"datetime_from": "timestamp"}).copy()

        # EU decimals to float
        for c in ["Upward", "Downward"]:
            if c in df.columns:
                df[c] = (
                    df[c]
                    .astype(str)
                    .str.replace("\u00a0", "", regex=False)  # NBSP
                    .str.replace(" ", "", regex=False)
                    .str.replace(",", ".", regex=False)
                    .replace({"": None})
                    .astype(float)
                )

        # Pick a single price series
        if {"Upward", "Downward"} <= set(df.columns) and (
            df["Upward"] - df["Downward"]
        ).abs().fillna(0).max() < 1e-9:
            df["price"] = df["Upward"]
        elif {"Upward", "Downward"} <= set(df.columns):
            df["price"] = df[["Upward", "Downward"]].mean(axis=1)
        elif "Target" in df.columns:  # in case the CSV exposes a Target column
            df["price"] = pd.to_numeric(df["Target"], errors="coerce")
        else:
            raise ValueError(
                "Could not infer a 'price' column from Upward/Downward/Target"
            )

        # Local market time (Baltics share rules); use a single IANA zone for joins
        df["timestamp"] = df["timestamp"].dt.tz_localize(
            tz, nonexistent="shift_forward", ambiguous="NaT"
        )
        df["timestamp"] = df["timestamp"].dt.floor("h")

        cols = ["timestamp", "area", "price", "Upward", "Downward", "datetime_to"]
        cols = [c for c in cols if c in df.columns]
        return df[cols].sort_values(["area", "timestamp"])

    def fetch_and_clean(self) -> pd.DataFrame:
        """
        One-shot helper to fetch CSV and return a cleaned DataFrame.
        """
        df_raw = self.fetch_balancing_prices()
        return self.clean_balancing_df(df_raw, self.tz_market)

    def make_target_hourly(
        self, df_clean: pd.DataFrame, tol_eur: float = 0.01, fill_limit: int = 3
    ) -> pd.Series:
        start_ts = pd.Timestamp(f"{self.start_date} 00:00", tz=self.tz_market)
        end_excl = pd.Timestamp(
            f"{self.end_date} 00:00", tz=self.tz_market
        ) + pd.Timedelta(days=1)

        # hourly grid [start, end)
        idx = pd.date_range(start=start_ts, end=end_excl, freq="h", inclusive="left")

        cols = {c.lower(): c for c in df_clean.columns}
        val_cols = [
            c
            for c in (
                cols.get("target"),
                cols.get("price"),
                cols.get("upward"),
                cols.get("downward"),
            )
            if c
        ]
        if not val_cols:
            raise ValueError("No target/price/upward/downward columns found.")

        long_vals = (
            df_clean[["timestamp"] + val_cols]
            .melt(id_vars=["timestamp"], value_vars=val_cols, value_name="v")
            .dropna(subset=["v"])
            .assign(v=lambda d: pd.to_numeric(d["v"], errors="coerce"))
            .dropna(subset=["v"])
        )

        # optional quick QC
        spread = long_vals.groupby("timestamp")["v"].agg(lambda s: s.max() - s.min())
        disag = int((spread > tol_eur).sum())
        if disag:
            print(
                f"[target] hours with >{tol_eur} EUR spread across areas/dirs: {disag}"
            )

        y = (
            long_vals.groupby("timestamp")["v"]
            .median()
            .rename("y")
            .sort_index()
            .reindex(idx)
            .interpolate(limit=fill_limit, limit_direction="both")
        )
        return y

    def make_origin_features(
        self,
        y: pd.Series,
        *,
        lags=(1, 8, 12, 24, 168),
        roll_windows=(24, 168),
        add_origin_calendar=True,
        prefix="fe_bd_",
    ) -> pd.DataFrame:
        """
        Origin-time features from the TARGET series (strictly past).
        Returns a DataFrame indexed by timestamp, columns prefixed with `prefix`.
        """
        X = pd.DataFrame(index=y.index)
        # lags (strictly past)
        for lag in lags:
            X[f"{prefix}lag_{lag}"] = y.shift(lag)

        # rolling stats (strictly past → shift(1))
        for w in roll_windows:
            X[f"{prefix}roll_mean_{w}"] = y.rolling(w).mean().shift(1)
            X[f"{prefix}roll_std_{w}"] = y.rolling(w).std(ddof=0).shift(1)

        if add_origin_calendar:
            X[f"{prefix}origin_hour"] = X.index.hour
            X[f"{prefix}origin_dow"] = X.index.dayofweek
            X[f"{prefix}origin_is_weekend"] = (X[f"{prefix}origin_dow"] >= 5).astype(
                int
            )

        return X

    def make_target_multihorizon(
        self,
        y_df: pd.DataFrame,  # output of make_target_hourly
        horizons: Iterable[int] = range(24),
        prefix: str = "target_",
    ) -> pd.DataFrame:
        """
        Create multi-horizon targets from hourly series.
        - One column per horizon (0h to 23h ahead)
        - Safe for supervised ML — no leakage

        Parameters
        ----------
        y_df : pd.DataFrame
            Must have index = hourly timestamp and a 'y' column (target).
        horizons : Iterable[int]
            Horizons to predict, in hours ahead of current time t0.
        prefix : str
            Column name prefix for targets.

        Returns
        -------
        pd.DataFrame
            index = timestamp (t0), columns = one per horizon: target_0, ..., target_23
        """
        if "y" not in y_df.columns:
            raise ValueError("Expected a column 'y' in input.")

        y_df = y_df.sort_index()
        targets = {}
        for h in horizons:
            targets[f"{prefix}{h}"] = y_df["y"].shift(-h)

        return pd.DataFrame(targets, index=y_df.index)

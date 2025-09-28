import requests
import pandas as pd
from typing import Dict, Optional
from datetime import datetime, timezone
import xml.etree.ElementTree as ET


class EntsoData:

    ENTSOE_BASE = "https://web-api.tp.entsoe.eu/api"

    # Bidding-zone EIC codes
    BALTIC_EIC = {
        "Estonia": "10Y1001A1001A39I",
        "Latvia": "10YLV-1001A00074",
        "Lithuania": "10YLT-1001A0008Q",
    }

    def __init__(
        self,
        api_key: str,
        start_date: str = None,
        end_date: str = None,
        tz: str = "Europe/Riga",
    ):
        self.api_key = api_key
        self.tz = tz
        self.start_date = start_date
        self.end_date = end_date

    @staticmethod
    def _utc_compact(dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y%m%d%H%M")

    def fetch_entsoe_day_ahead_prices(
        self,
        eic_code: str,
        start_utc: datetime,
        end_utc: datetime,
    ) -> pd.DataFrame:

        params = {
            "securityToken": self.api_key,
            "documentType": "A44",
            "in_Domain": eic_code,
            "out_Domain": eic_code,
            "periodStart": self._utc_compact(start_utc),
            "periodEnd": self._utc_compact(end_utc),
        }
        r = requests.get(self.ENTSOE_BASE, params=params, timeout=60)
        if r.status_code >= 400:
            # show the server’s message — ENT SO-E tells you if the window is too long, etc.
            msg = r.text[:800]
            raise requests.HTTPError(
                f"ENTSO-E {r.status_code} for {eic_code}: {msg}", response=r
            )
        root = ET.fromstring(r.text)

        ns = {"ns": root.tag.split("}")[0].strip("{")}
        currency = None
        for el in root.findall(".//ns:currency_Unit.name", ns) + root.findall(
            ".//ns:price_Measure_Unit.name", ns
        ):
            if el.text:
                currency = el.text
                break

        records = []
        for ts in root.findall(".//ns:TimeSeries", ns):
            for period in ts.findall(".//ns:Period", ns):
                t0_el = period.find("./ns:timeInterval/ns:start", ns)
                res_el = period.find("./ns:resolution", ns)
                if t0_el is None:
                    continue
                t0 = pd.to_datetime(t0_el.text, utc=True)
                step = (
                    pd.to_timedelta(res_el.text.replace("PT", "").lower())
                    if res_el is not None
                    else pd.Timedelta("1h")
                )
                for p in period.findall("./ns:Point", ns):
                    pos = int(p.find("./ns:position", ns).text)
                    price = float(p.find("./ns:price.amount", ns).text)
                    ts_utc = t0 + (pos - 1) * step
                    records.append((ts_utc, price))

        if not records:
            # Don’t 400, but empty payload → keep it explicit
            raise ValueError(
                f"Empty ENTSO-E response for {eic_code}. First 400 chars:\n{r.text[:400]}"
            )

        df = pd.DataFrame(records, columns=["timestamp_utc", "price"]).sort_values(
            "timestamp_utc"
        )
        df["timestamp"] = df["timestamp_utc"].dt.tz_convert(self.tz)
        df = df.drop(columns="timestamp_utc").set_index("timestamp").asfreq("h")
        df["currency"] = currency or "EUR"
        return df

    def fetch_baltic_day_ahead_prices(
        self,
        area_eic: Optional[Dict[str, str]] = None,
        max_days_per_call: int = 365,  # <= 1 year per API call
    ) -> pd.DataFrame:
        area_eic = area_eic or self.BALTIC_EIC

        # build local endpoints
        start_local = pd.Timestamp(f"{self.start_date} 00:00", tz=self.tz)
        end_local_excl = pd.Timestamp(
            f"{self.end_date} 00:00", tz=self.tz
        ) + pd.Timedelta(days=1)

        frames = []
        cur = start_local
        while cur < end_local_excl:
            chunk_end_local = min(
                cur + pd.Timedelta(days=max_days_per_call), end_local_excl
            )
            s_utc = cur.tz_convert("UTC").to_pydatetime()
            e_utc = chunk_end_local.tz_convert("UTC").to_pydatetime()

            for area, eic in area_eic.items():
                df = self.fetch_entsoe_day_ahead_prices(eic, s_utc, e_utc)
                df = df.rename(columns={"price": "da_price_eur"}).reset_index()
                df["area"] = area
                frames.append(df[["timestamp", "area", "da_price_eur"]])

            cur = chunk_end_local  # advance

        out = pd.concat(frames, ignore_index=True).sort_values(["area", "timestamp"])
        # guard against overlap duplicates (at chunk boundaries)
        out = out.drop_duplicates(["area", "timestamp"])
        return out

    def make_dayahead_hourly(
        self, df_da: pd.DataFrame, fill_limit: int = 3
    ) -> pd.DataFrame:
        start_ts = pd.Timestamp(f"{self.start_date} 00:00", tz=self.tz)
        end_excl = pd.Timestamp(f"{self.end_date} 00:00", tz=self.tz) + pd.Timedelta(
            days=1
        )
        idx = pd.date_range(start=start_ts, end=end_excl, freq="h", inclusive="left")

        if "da_price_eur" not in df_da.columns:
            raise ValueError("df_da must contain 'da_price_eur'.")

        da = (
            df_da.groupby("timestamp")["da_price_eur"]
            .agg(da_mean="mean")
            .sort_index()
            .reindex(idx)
        )
        da = da.interpolate(limit=fill_limit)
        return da

    def make_dayahead_features_full(
        self,
        df: pd.DataFrame,
        *,
        column: str = "da_mean",
        lags=(0, 1, 24),
        rolls=(24, 168),
        diffs=(1, 24),
        prefix="fe_ed_",
    ) -> pd.DataFrame:
        """
        Safe origin-time features from day-ahead prices.

        Parameters:
            da: DataFrame with datetime index and a day-ahead price column
            column: Name of the day-ahead price column (default: "da_mean")
            lags: Time-based lags to include (in hours)
            rolls: Rolling windows for mean/std (applied with shift(1))
            diffs: Differences to previous values (e.g., diff_24 = x - x.shift(24))
            prefix: Prefix for feature names

        Returns:
            DataFrame with origin-time features.
        """
        df = df.sort_index().copy()
        x = df[column]
        out = {}

        for L in lags:
            out[f"{prefix}lag_{L}"] = x.shift(L)

        for W in rolls:
            out[f"{prefix}roll_mean_{W}"] = x.rolling(W).mean().shift(1)
            out[f"{prefix}roll_std_{W}"] = x.rolling(W).std(ddof=0).shift(1)

        for D in diffs:
            out[f"{prefix}diff_{D}"] = x - x.shift(D)

        return pd.DataFrame(out, index=df.index)

import sys

sys.path.append("BalancingEnergyPrice")

import pandas as pd
from pathlib import Path

# Import custom modules
import params as p
from modules.utils import safe_merge_many

from modules.balticData import BalticData
from modules.mateoData import MateoData
from modules.entsoData import EntsoData

# ----------------- PARAMETERS -----------------
# derive historical train window
HIST_START = (
    (pd.to_datetime(p.START) - pd.DateOffset(years=p.history_years)).date().isoformat()
)
TRAIN_END = (pd.to_datetime(p.START) - pd.DateOffset(days=1)).date().isoformat()

print(
    "Fetch window:",
    HIST_START,
    "→",
    p.END,
    "| Train:",
    HIST_START,
    "→",
    TRAIN_END,
    "| Test:",
    p.START,
    "→",
    p.END,
)

# ----------------- FETCH DATA -----------------
# Balancing prices (BTD)
bd = BalticData(
    start_date=HIST_START,
    end_date=p.END,
    tz_api="EET",
    tz_market=p.TZ,
    out_dir=Path(p.output_dir),
    out_name="baltic_prices_" + str(p.history_years) + "y_plus_sep2024.csv",
)
bd_df = bd.fetch_and_clean()

# Day-ahead (ENTSO-E)
ed = EntsoData(api_key=p.API_TOKEN, start_date=HIST_START, end_date=p.END, tz=p.TZ)
ed_df = ed.fetch_baltic_day_ahead_prices()

# Weather (Meteo)
weather_vars = [
    "temperature_2m",
    "windspeed_10m",
    "shortwave_radiation",
    "cloudcover",
    "pressure_msl",
]

md = MateoData(
    start_date=HIST_START,
    end_date=p.END,
    tz=p.TZ,
    area_points=p.AREA_POINTS,
    hourly_vars=weather_vars,
    weights=p.WEIGHTS,
    agg="mean",
)
wx_df = md.build_area_weather_multi()

# ----------------- ONE ROW PER HOUR (market-level) -----------------
# Target across the full window
y_all = bd.make_target_hourly(bd_df)
Y_all = bd.make_target_multihorizon(y_all.to_frame(name="y"), horizons=range(24))

Xbd_all = bd.make_origin_features(y_all)

# Day-ahead features across the full window
Xed_all = ed.make_dayahead_hourly(ed_df)
Xed_all = ed.make_dayahead_features_full(Xed_all)

# Weather features across the full window
Xwx_all = md.make_weather_hourly(wx_df)
Xwx_all = md.make_weather_features(Xwx_all)

# ----------------- MERGE -----------------
# Merge all features by index
X_all = safe_merge_many([Xbd_all, Xed_all, Xwx_all])

# ------------------ DAILY DATA -----------------
# Take only rows at midnight (00:00) for daily model
X_daily = X_all[X_all.index.hour == 0]
Y_daily = Y_all[Y_all.index.hour == 0]

# ------------------ SAVE DATA -----------------

X_daily.to_csv(Path(p.output_dir) / "X_daily.csv")
Y_daily.to_csv(Path(p.output_dir) / "Y_daily.csv")

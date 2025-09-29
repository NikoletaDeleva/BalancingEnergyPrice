# Data window
START = "2024-09-01"
END = "2024-09-30"
history_years = 5  # how many years of history to use for training
output_dir = "G:/Niki/Data"
TZ = "Europe/Riga"

# Weather area points + weights
# (3 points per country, roughly covering areas - could choose areas that corespond to the areas with most usage)
AREA_POINTS = {
    "Estonia": [(59.44, 24.75), (58.38, 26.73), (58.98, 22.50)],
    "Latvia": [(56.95, 24.11), (57.39, 21.56), (56.65, 25.25)],
    "Lithuania": [(54.69, 25.28), (55.70, 21.13), (55.08, 23.32)],
}

# Weather area weights (to average the points into a single country-level value)
# (could be adjusted based on population density, usage, etc - here just rough estimates)
WEIGHTS = {
    "Estonia": [0.55, 0.30, 0.15],
    "Latvia": [0.60, 0.25, 0.15],
    "Lithuania": [0.55, 0.25, 0.20],
}

# ENTSO-E API token (free registration at https://transparency.entsoe.eu/ and email them for a token)
API_TOKEN = ""

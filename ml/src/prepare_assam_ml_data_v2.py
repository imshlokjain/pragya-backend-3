import pandas as pd
import os

RAIN_FILE = "ml/data/assam/processed/assam_rainfall_clean.csv"
RIVER_FILE = "ml/data/assam/processed/assam_river_clean.csv"

OUTPUT_FILE = "ml/data/assam/processed/assam_ml_dataset_v2.csv"

print("=" * 70)
print("LOADING CLEANED DATA")
print("=" * 70)

rain = pd.read_csv(RAIN_FILE)
river = pd.read_csv(RIVER_FILE)

rain["timestamp"] = pd.to_datetime(rain["timestamp"])
river["timestamp"] = pd.to_datetime(river["timestamp"])

print("Rainfall records:", len(rain))
print("River records:", len(river))


# ============================================================
# KEEP COMMON STATIONS
# ============================================================

common_stations = sorted(
    set(rain["Station"]).intersection(set(river["Station"]))
)

print("\n" + "=" * 70)
print("COMMON STATIONS")
print("=" * 70)

for station in common_stations:
    print("-", station)

rain = rain[rain["Station"].isin(common_stations)].copy()
river = river[river["Station"].isin(common_stations)].copy()

print("\nRainfall records:", len(rain))
print("River records:", len(river))


# ============================================================
# REMOVE DUPLICATE RAINFALL OBSERVATIONS
# ============================================================

print("\n" + "=" * 70)
print("AGGREGATING DUPLICATES")
print("=" * 70)

rain = (
    rain.groupby(
        [
            "Station",
            "District",
            "Latitude",
            "Longitude",
            "timestamp",
        ],
        as_index=False
    )
    .agg(
        rainfall_mm=("rainfall_mm", "sum")
    )
)

print("Rainfall records after aggregation:", len(rain))


# ============================================================
# MERGE RIVER LEVEL WITH RAINFALL OBSERVATIONS
# ============================================================

print("\n" + "=" * 70)
print("MATCHING RIVER LEVEL TO RAINFALL OBSERVATIONS")
print("=" * 70)

# merge_asof requires sorting primarily by timestamp
rain = rain.sort_values(
    ["timestamp", "Station"]
)

river = river.sort_values(
    ["timestamp", "Station"]
)

df = pd.merge_asof(
    rain,
    river[
        [
            "Station",
            "timestamp",
            "river_level",
        ]
    ],
    on="timestamp",
    by="Station",
    direction="nearest",
    tolerance=pd.Timedelta("2h"),
)

print("Records after merge:", len(df))

missing_river = df["river_level"].isna().sum()

print("Missing river matches:", missing_river)

# Remove observations without nearby river measurement
df = df.dropna(subset=["river_level"]).copy()

print("Records with both rainfall + river data:", len(df))


# ============================================================
# SORT FOR TIME SERIES FEATURES
# ============================================================

df = df.sort_values(
    ["Station", "timestamp"]
).reset_index(drop=True)


# ============================================================
# TIME FEATURES
# ============================================================

print("\n" + "=" * 70)
print("CREATING FEATURES")
print("=" * 70)

df["month"] = df["timestamp"].dt.month
df["day_of_year"] = df["timestamp"].dt.dayofyear


# ============================================================
# RAINFALL FEATURES
# ============================================================

g = df.groupby("Station", group_keys=False)

# Previous observed rainfall
df["rainfall_previous"] = g["rainfall_mm"].shift(1).fillna(0)

# Rolling rainfall based on recent observations
df["rainfall_3_obs"] = g["rainfall_mm"].transform(
    lambda x: x.rolling(3, min_periods=1).sum()
)

df["rainfall_6_obs"] = g["rainfall_mm"].transform(
    lambda x: x.rolling(6, min_periods=1).sum()
)


# ============================================================
# RIVER FEATURES
# ============================================================

df["river_level_change"] = g["river_level"].diff().fillna(0)

df["river_level_rolling_mean_3"] = g["river_level"].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)

df["river_level_rolling_max_6"] = g["river_level"].transform(
    lambda x: x.rolling(6, min_periods=1).max()
)


# ============================================================
# FINAL DATASET
# ============================================================

columns = [
    "Station",
    "District",
    "Latitude",
    "Longitude",
    "timestamp",
    "month",
    "day_of_year",
    "rainfall_mm",
    "rainfall_previous",
    "rainfall_3_obs",
    "rainfall_6_obs",
    "river_level",
    "river_level_change",
    "river_level_rolling_mean_3",
    "river_level_rolling_max_6",
]

df = df[columns]

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print("\nShape:", df.shape)

print("\nDate range:")
print(df["timestamp"].min(), "to", df["timestamp"].max())

print("\nRecords per station:")
print(df.groupby("Station").size())

print("\nMissing values:")
print(df.isna().sum())

print("\nSaved successfully:")
print(OUTPUT_FILE)

print("\nFirst 10 rows:")
print(df.head(10).to_string(index=False))

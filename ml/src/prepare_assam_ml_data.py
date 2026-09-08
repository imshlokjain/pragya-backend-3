import pandas as pd
import os

RAIN_FILE = "ml/data/assam/processed/assam_rainfall_clean.csv"
RIVER_FILE = "ml/data/assam/processed/assam_river_clean.csv"

OUTPUT_FILE = "ml/data/assam/processed/assam_ml_dataset_clean.csv"

print("=" * 70)
print("LOADING CLEANED DATA")
print("=" * 70)

rain = pd.read_csv(RAIN_FILE)
river = pd.read_csv(RIVER_FILE)

rain["timestamp"] = pd.to_datetime(rain["timestamp"])
river["timestamp"] = pd.to_datetime(river["timestamp"])

print("Rainfall records:", len(rain))
print("River records:", len(river))

print("\nRainfall stations:", rain["Station"].nunique())
print("River stations:", river["Station"].nunique())


# --------------------------------------------------
# KEEP ONLY COMMON STATIONS
# --------------------------------------------------

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

print("\nRainfall records after filtering:", len(rain))
print("River records after filtering:", len(river))


# --------------------------------------------------
# AGGREGATE RAINFALL BY STATION + HOUR
# --------------------------------------------------

print("\n" + "=" * 70)
print("AGGREGATING RAINFALL")
print("=" * 70)

rain["hour"] = rain["timestamp"].dt.floor("h")

rain_hourly = (
    rain.groupby(["Station", "hour"], as_index=False)
    .agg(
        rainfall_mm=("rainfall_mm", "sum"),
        District=("District", "first"),
        Latitude=("Latitude", "first"),
        Longitude=("Longitude", "first"),
    )
    .rename(columns={"hour": "timestamp"})
)

print("Hourly rainfall records:", len(rain_hourly))


# --------------------------------------------------
# MERGE WITH RIVER DATA
# --------------------------------------------------

print("\n" + "=" * 70)
print("MERGING DATA")
print("=" * 70)

# merge_asof requires global timestamp sorting
rain_hourly = rain_hourly.sort_values(
    ["timestamp", "Station"]
)

river = river.sort_values(
    ["timestamp", "Station"]
)

df = pd.merge_asof(
    river,
    rain_hourly[
        [
            "Station",
            "timestamp",
            "rainfall_mm",
        ]
    ],
    on="timestamp",
    by="Station",
    direction="nearest",
    tolerance=pd.Timedelta("1h"),
)

print("Records after merge:", len(df))
print("Missing rainfall matches:", df["rainfall_mm"].isna().sum())

# Missing rainfall means no rainfall telemetry near that time.
# Treat it as zero observation for the hourly feature.
df["rainfall_mm"] = df["rainfall_mm"].fillna(0)


# --------------------------------------------------
# SORT FOR TIME-SERIES FEATURES
# --------------------------------------------------

df = df.sort_values(
    ["Station", "timestamp"]
).reset_index(drop=True)


# --------------------------------------------------
# CREATE TIME-SERIES FEATURES
# --------------------------------------------------

print("\n" + "=" * 70)
print("CREATING TIME-SERIES FEATURES")
print("=" * 70)

g = df.groupby("Station", group_keys=False)

# Rainfall accumulation
df["rainfall_6h"] = g["rainfall_mm"].transform(
    lambda x: x.rolling(6, min_periods=1).sum()
)

df["rainfall_24h"] = g["rainfall_mm"].transform(
    lambda x: x.rolling(24, min_periods=1).sum()
)

# River level changes
df["river_level_change_1h"] = g["river_level"].diff(1).fillna(0)

df["river_level_change_6h"] = g["river_level"].diff(6).fillna(0)

df["river_level_change_24h"] = g["river_level"].diff(24).fillna(0)

# River rolling statistics
df["river_level_mean_6h"] = g["river_level"].transform(
    lambda x: x.rolling(6, min_periods=1).mean()
)

df["river_level_max_24h"] = g["river_level"].transform(
    lambda x: x.rolling(24, min_periods=1).max()
)


# --------------------------------------------------
# FINAL CLEANUP
# --------------------------------------------------

df = df.sort_values(
    ["Station", "timestamp"]
).reset_index(drop=True)

columns = [
    "Station",
    "District",
    "Latitude",
    "Longitude",
    "timestamp",
    "river_level",
    "rainfall_mm",
    "rainfall_6h",
    "rainfall_24h",
    "river_level_change_1h",
    "river_level_change_6h",
    "river_level_change_24h",
    "river_level_mean_6h",
    "river_level_max_24h",
]

df = df[columns]

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)


print("\n" + "=" * 70)
print("FINAL CLEAN DATASET")
print("=" * 70)

print("\nShape:", df.shape)

print("\nMissing values:")
print(df.isna().sum())

print("\nDate range:")
print(df["timestamp"].min(), "to", df["timestamp"].max())

print("\nSaved successfully:")
print(OUTPUT_FILE)

print("\nFirst 10 rows:")
print(df.head(10).to_string(index=False))

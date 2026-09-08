import os
import pandas as pd
import numpy as np


RAINFALL_FILE = "ml/data/assam/raw/assam_rainfall.csv"
RIVER_FILE = "ml/data/assam/raw/assam_river_level.csv"

OUTPUT_FILE = "ml/data/assam/processed/assam_ml_dataset.csv"


print("=" * 70)
print("LOADING ASSAM RAINFALL DATA")
print("=" * 70)

rain = pd.read_csv(RAINFALL_FILE)

rain["timestamp"] = pd.to_datetime(
    rain["Data Acquisition Time"],
    format="%d-%m-%Y %H:%M",
    errors="coerce",
)

rain = rain.rename(
    columns={
        "Telemetry Hourly Rainfall (mm)": "rainfall_mm"
    }
)

rain = rain[
    [
        "Station",
        "District",
        "Latitude",
        "Longitude",
        "timestamp",
        "rainfall_mm",
    ]
].copy()

rain = rain.dropna(
    subset=["timestamp", "rainfall_mm"]
)

print("Rainfall records:", len(rain))


print("\n" + "=" * 70)
print("LOADING ASSAM RIVER LEVEL DATA")
print("=" * 70)

river = pd.read_csv(RIVER_FILE)

river["timestamp"] = pd.to_datetime(
    river["Data Acquisition Time"],
    format="%d-%m-%Y %H:%M",
    errors="coerce",
)

river = river.rename(
    columns={
        "River Water Level Telemetry Hourly (meter)": "river_level"
    }
)

river = river[
    [
        "Station",
        "District",
        "Latitude",
        "Longitude",
        "timestamp",
        "river_level",
    ]
].copy()

river = river.dropna(
    subset=["timestamp", "river_level"]
)

print("River records:", len(river))


print("\n" + "=" * 70)
print("KEEPING COMMON STATIONS ONLY")
print("=" * 70)

common_stations = sorted(
    set(rain["Station"]).intersection(
        set(river["Station"])
    )
)

print("Common stations:")

for station in common_stations:
    print("-", station)

rain = rain[
    rain["Station"].isin(common_stations)
].copy()

river = river[
    river["Station"].isin(common_stations)
].copy()


print("\nRainfall records after filtering:", len(rain))
print("River records after filtering:", len(river))


print("\n" + "=" * 70)
print("MERGING RAINFALL AND RIVER DATA")
print("=" * 70)


rain = rain.sort_values(
    ["timestamp", "Station"]
)

river = river.sort_values(
    ["timestamp", "Station"]
)


# Merge observations using station and timestamp
df = pd.merge_asof(
    river,
    rain[
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


df = df.sort_values(
    ["timestamp", "Station"]
).reset_index(drop=True)


print("Merged records:", len(df))

print("Missing rainfall after merge:")

print(df["rainfall_mm"].isna().sum())


# Missing rainfall means no rainfall observation
# in the matching time window.
# We treat it as 0 only for the ML feature pipeline.

df["rainfall_mm"] = df[
    "rainfall_mm"
].fillna(0)


print("\n" + "=" * 70)
print("CREATING TIME-SERIES FEATURES")
print("=" * 70)


df = df.sort_values(
    ["timestamp", "Station"]
).reset_index(drop=True)


# Rainfall accumulation features
df["rainfall_6h"] = (
    df.groupby("Station")["rainfall_mm"]
    .transform(
        lambda x: x.rolling(
            6,
            min_periods=1
        ).sum()
    )
)


df["rainfall_24h"] = (
    df.groupby("Station")["rainfall_mm"]
    .transform(
        lambda x: x.rolling(
            24,
            min_periods=1
        ).sum()
    )
)


# River-level changes
df["river_level_change_1h"] = (
    df.groupby("Station")["river_level"]
    .diff(1)
)


df["river_level_change_6h"] = (
    df.groupby("Station")["river_level"]
    .diff(6)
)


df["river_level_change_24h"] = (
    df.groupby("Station")["river_level"]
    .diff(24)
)


# Rolling river statistics
df["river_level_mean_6h"] = (
    df.groupby("Station")["river_level"]
    .transform(
        lambda x: x.rolling(
            6,
            min_periods=1
        ).mean()
    )
)


df["river_level_max_24h"] = (
    df.groupby("Station")["river_level"]
    .transform(
        lambda x: x.rolling(
            24,
            min_periods=1
        ).max()
    )
)


# Fill initial differences
change_columns = [
    "river_level_change_1h",
    "river_level_change_6h",
    "river_level_change_24h",
]

df[change_columns] = df[
    change_columns
].fillna(0)


print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print("Shape:", df.shape)

print("\nColumns:")

print(df.columns.tolist())


os.makedirs(
    "ml/data/assam/processed",
    exist_ok=True,
)


df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print("\nSaved successfully:")

print(OUTPUT_FILE)


print("\nFirst 10 rows:")

print(
    df.head(10).to_string(
        index=False
    )
)


print("\nMissing values:")

print(
    df.isnull().sum()
)

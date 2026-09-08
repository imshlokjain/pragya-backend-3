import pandas as pd
import os

INPUT_FILE = "ml/data/assam/raw/assam_river_level.csv"
OUTPUT_FILE = "ml/data/assam/processed/assam_river_clean.csv"

COL = "River Water Level Telemetry Hourly (meter)"

print("=" * 60)
print("LOADING RIVER DATA")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("Original records:", len(df))
print("Missing values:", df[COL].isna().sum())

# Remove missing values
df = df.dropna(subset=[COL]).copy()

# Remove negative values if any
df = df[df[COL] >= 0].copy()

print("\nRecords after removing missing/negative:", len(df))

print("\n" + "=" * 60)
print("STATION-WISE OUTLIER CLEANING")
print("=" * 60)

cleaned_parts = []

for station, group in df.groupby("Station"):
    threshold = group[COL].quantile(0.999)

    before = len(group)

    clean_group = group[group[COL] <= threshold].copy()

    removed = before - len(clean_group)

    print(f"\n{station}")
    print(f"99.9% threshold: {threshold:.3f} m")
    print(f"Records before: {before}")
    print(f"Removed: {removed}")
    print(f"Records after: {len(clean_group)}")

    cleaned_parts.append(clean_group)

cleaned = pd.concat(cleaned_parts, ignore_index=True)

# Rename useful columns
cleaned = cleaned.rename(
    columns={
        "Data Acquisition Time": "timestamp",
        COL: "river_level"
    }
)

# Convert timestamp
cleaned["timestamp"] = pd.to_datetime(
    cleaned["timestamp"],
    dayfirst=True,
    errors="coerce"
)

# Remove invalid timestamps
cleaned = cleaned.dropna(subset=["timestamp"])

# Sort
cleaned = cleaned.sort_values(
    ["Station", "timestamp"]
).reset_index(drop=True)

# Keep useful columns
columns_to_keep = [
    "Station",
    "District",
    "Latitude",
    "Longitude",
    "timestamp",
    "river_level"
]

cleaned = cleaned[columns_to_keep]

# Create directory if needed
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

cleaned.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 60)
print("CLEANING COMPLETE")
print("=" * 60)

print("\nFinal records:", len(cleaned))

print("\nDate range:")
print(cleaned["timestamp"].min(), "to", cleaned["timestamp"].max())

print("\nFinal statistics:")
print(cleaned["river_level"].describe())

print("\nSaved to:")
print(OUTPUT_FILE)

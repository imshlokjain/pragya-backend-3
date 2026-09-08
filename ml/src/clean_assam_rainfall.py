import pandas as pd
import os

INPUT_FILE = "ml/data/assam/raw/assam_rainfall.csv"
OUTPUT_FILE = "ml/data/assam/processed/assam_rainfall_clean.csv"

RAIN_COL = "Telemetry Hourly Rainfall (mm)"

print("=" * 70)
print("LOADING ASSAM RAINFALL DATA")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("Original records:", len(df))

# Convert timestamp
df["timestamp"] = pd.to_datetime(
    df["Data Acquisition Time"],
    dayfirst=True,
    errors="coerce"
)

# Convert rainfall to numeric
df[RAIN_COL] = pd.to_numeric(
    df[RAIN_COL],
    errors="coerce"
)

print("\nMissing timestamps:", df["timestamp"].isna().sum())
print("Missing rainfall:", df[RAIN_COL].isna().sum())

print("\nRemoving invalid rainfall values...")

# Remove negative rainfall
df = df[df[RAIN_COL] >= 0]

# Remove clearly corrupted values
df = df[df[RAIN_COL] <= 1000]

print("Records after cleaning:", len(df))

# Remove duplicate station + timestamp observations
before = len(df)

df = df.drop_duplicates(
    subset=["Station", "timestamp"],
    keep="last"
)

print("Duplicates removed:", before - len(df))

# Keep useful columns
clean_df = df[
    [
        "Station",
        "District",
        "Latitude",
        "Longitude",
        "timestamp",
        RAIN_COL,
    ]
].copy()

clean_df = clean_df.rename(
    columns={
        RAIN_COL: "rainfall_mm"
    }
)

# Sort properly
clean_df = clean_df.sort_values(
    ["Station", "timestamp"]
)

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

clean_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("CLEANING COMPLETE")
print("=" * 70)

print("\nFinal records:", len(clean_df))

print("\nDate range:")
print(
    clean_df["timestamp"].min(),
    "to",
    clean_df["timestamp"].max()
)

print("\nRainfall statistics:")
print(
    clean_df["rainfall_mm"].describe()
)

print("\nSaved to:")
print(OUTPUT_FILE)

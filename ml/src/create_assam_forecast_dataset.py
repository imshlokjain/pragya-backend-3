import pandas as pd
import numpy as np
import os

INPUT_FILE = "ml/data/assam/processed/assam_ml_dataset_v2.csv"
OUTPUT_FILE = "ml/data/assam/processed/assam_forecast_dataset.csv"

FORECAST_HOURS = 6
MAX_GAP_HOURS = 12
TOLERANCE_HOURS = 2


print("=" * 70)
print("LOADING ASSAM ML DATA")
print("=" * 70)

df = pd.read_csv(
    INPUT_FILE,
    parse_dates=["timestamp"]
)

df = df.sort_values(["Station", "timestamp"]).reset_index(drop=True)

print("Records:", len(df))


print("\n" + "=" * 70)
print("CREATING CONTINUOUS TIME SEQUENCES")
print("=" * 70)

df["time_gap_hours"] = (
    df.groupby("Station")["timestamp"]
      .diff()
      .dt.total_seconds()
      .div(3600)
)

df["new_sequence"] = (
    (df["time_gap_hours"].isna()) |
    (df["time_gap_hours"] > MAX_GAP_HOURS)
)

df["sequence_id"] = (
    df.groupby("Station")["new_sequence"]
      .cumsum()
)

print("Sequences created:", df.groupby(["Station", "sequence_id"]).ngroups)


print("\n" + "=" * 70)
print("CREATING 6-HOUR FUTURE TARGET")
print("=" * 70)

all_results = []

for station, group in df.groupby("Station"):

    group = group.sort_values("timestamp").copy()

    left = group.copy()

    left["target_timestamp"] = (
        left["timestamp"] +
        pd.Timedelta(hours=FORECAST_HOURS)
    )

    right = group[
        ["timestamp", "river_level", "sequence_id"]
    ].copy()

    right = right.rename(
        columns={
            "timestamp": "future_timestamp",
            "river_level": "future_river_level",
            "sequence_id": "future_sequence_id",
        }
    )

    left = left.sort_values("target_timestamp")
    right = right.sort_values("future_timestamp")

    merged = pd.merge_asof(
        left,
        right,
        left_on="target_timestamp",
        right_on="future_timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(hours=TOLERANCE_HOURS),
    )

    all_results.append(merged)


forecast_df = pd.concat(all_results, ignore_index=True)

print("Records before target filtering:", len(forecast_df))


print("\n" + "=" * 70)
print("FILTERING VALID FUTURE TARGETS")
print("=" * 70)

forecast_df = forecast_df.dropna(
    subset=[
        "future_timestamp",
        "future_river_level",
    ]
)

forecast_df = forecast_df[
    forecast_df["sequence_id"] ==
    forecast_df["future_sequence_id"]
]

print("Records with valid future targets:", len(forecast_df))


forecast_df["actual_forecast_gap_hours"] = (
    forecast_df["future_timestamp"] -
    forecast_df["timestamp"]
).dt.total_seconds() / 3600


forecast_df["future_river_level_change"] = (
    forecast_df["future_river_level"] -
    forecast_df["river_level"]
)


print("\n" + "=" * 70)
print("FINAL FORECAST DATASET")
print("=" * 70)

print("\nShape:", forecast_df.shape)

print("\nForecast gap statistics:")
print(
    forecast_df["actual_forecast_gap_hours"]
    .describe()
)

print("\nFuture river level change:")
print(
    forecast_df["future_river_level_change"]
    .describe()
)


columns_to_drop = [
    "new_sequence",
    "future_sequence_id",
]

forecast_df = forecast_df.drop(
    columns=columns_to_drop,
    errors="ignore"
)


os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

forecast_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nSaved successfully:")
print(OUTPUT_FILE)


print("\nRecords per station:")
print(
    forecast_df.groupby("Station")
    .size()
)


print("\nFirst 10 rows:")
print(
    forecast_df.head(10).to_string(index=False)
)

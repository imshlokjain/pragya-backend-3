import pandas as pd
import os

INPUT_FILE = "ml/data/raw/flood_risk_india.csv"
OUTPUT_FILE = "ml/data/pragya_training_data.csv"

df = pd.read_csv(INPUT_FILE)

# --------------------------------------------------
# Create PRAGYA backend-compatible features
# --------------------------------------------------

pragya_df = pd.DataFrame()

# Rainfall features
pragya_df["rainfall_mm"] = df["Rainfall (mm)"]

# Prototype approximation of accumulated rainfall
pragya_df["rainfall_24h"] = (
    df["Rainfall (mm)"] * 1.25
).round(2)

# River features
pragya_df["river_level"] = df["Water Level (m)"]

# Approximate rate of change for prototype training
pragya_df["river_level_change"] = (
    df["River Discharge (m³/s)"] / 10000
).round(4)

# Prototype danger level
danger_level = 10.0

pragya_df["distance_to_danger"] = (
    danger_level - pragya_df["river_level"]
).round(2)

# Historical flood information
pragya_df["historical_flood_frequency"] = (
    df["Historical Floods"]
)

# Satellite/flood features
pragya_df["flood_affected_area"] = (
    df["Historical Floods"]
    * df["Rainfall (mm)"]
    / 100
).round(2)

pragya_df["flood_confidence"] = (
    0.5
    + (df["Humidity (%)"] / 200)
).clip(0, 1).round(3)

# Target
pragya_df["flood_occurred"] = df["Flood Occurred"]

# Save
os.makedirs("ml/data", exist_ok=True)

pragya_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("PRAGYA training dataset created successfully")
print()
print("Shape:", pragya_df.shape)
print()
print("Columns:")
print(pragya_df.columns.tolist())
print()
print("First 5 rows:")
print(pragya_df.head())

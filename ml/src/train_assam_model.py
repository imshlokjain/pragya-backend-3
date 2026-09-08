import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_FILE = "ml/data/assam/processed/assam_forecast_dataset.csv"
MODEL_FILE = "ml/models/assam_river_model.pkl"


print("=" * 70)
print("LOADING FORECAST DATASET")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print("Records:", len(df))
print("Columns:")
print(df.columns.tolist())


# --------------------------------------------------
# SELECT FEATURES AND TARGET
# --------------------------------------------------

features = [
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

target = "future_river_level"


# --------------------------------------------------
# SORT BY TIME
# --------------------------------------------------

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)


# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]


print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print("Training records:", len(train_df))
print("Testing records:", len(test_df))


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST MODEL")
print("=" * 70)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

predictions = model.predict(X_test)


# --------------------------------------------------
# BASELINE MODEL
# --------------------------------------------------

baseline_predictions = X_test["river_level"]

baseline_mae = mean_absolute_error(y_test, baseline_predictions)

baseline_rmse = mean_squared_error(
    y_test,
    baseline_predictions
) ** 0.5

baseline_r2 = r2_score(y_test, baseline_predictions)


print("\n" + "=" * 70)
print("BASELINE MODEL")
print("=" * 70)

print("Baseline: future river level = current river level")
print(f"MAE:  {baseline_mae:.4f} meters")
print(f"RMSE: {baseline_rmse:.4f} meters")
print(f"R²:   {baseline_r2:.4f}")


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

mae = mean_absolute_error(y_test, predictions)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(y_test, predictions)


print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

print(f"MAE:  {mae:.4f} meters")
print(f"RMSE: {rmse:.4f} meters")
print(f"R²:   {r2:.4f}")


# --------------------------------------------------
# STATION-WISE EVALUATION
# --------------------------------------------------

print("\n" + "=" * 70)
print("STATION-WISE MODEL EVALUATION")
print("=" * 70)

results = test_df[["Station", target]].copy()
results["prediction"] = predictions

for station, group in results.groupby("Station"):

    station_mae = mean_absolute_error(
        group[target],
        group["prediction"]
    )

    station_rmse = mean_squared_error(
        group[target],
        group["prediction"]
    ) ** 0.5

    station_r2 = r2_score(
        group[target],
        group["prediction"]
    )

    print(f"\n{station}")
    print("-" * 50)
    print(f"Records: {len(group)}")
    print(f"MAE:  {station_mae:.4f} m")
    print(f"RMSE: {station_rmse:.4f} m")
    print(f"R²:   {station_r2:.4f}")


# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

print(importance.to_string(index=False))


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

joblib.dump(
    {
        "model": model,
        "features": features
    },
    MODEL_FILE
)

print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(MODEL_FILE)

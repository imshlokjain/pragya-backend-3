import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_FILE = "ml/data/assam/processed/assam_forecast_dataset.csv"
MODEL_FILE = "ml/models/assam_river_model.pkl"

df = pd.read_csv(DATA_FILE)

model_data = joblib.load(MODEL_FILE)

model = model_data["model"]
features = model_data["features"]

X = df[features]
y = df["future_river_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

test_df = df.loc[X_test.index].copy()

predictions = model.predict(X_test)

test_df["actual"] = y_test
test_df["predicted"] = predictions
test_df["error"] = (
    test_df["actual"] - test_df["predicted"]
).abs()

print("=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print("MAE:", round(
    mean_absolute_error(test_df["actual"], test_df["predicted"]), 4
))

print("RMSE:", round(
    mean_squared_error(
        test_df["actual"],
        test_df["predicted"]
    ) ** 0.5, 4
))

print("R2:", round(
    r2_score(test_df["actual"], test_df["predicted"]), 4
))


print("\n" + "=" * 70)
print("STATION RESULTS WITH OUTLIERS")
print("=" * 70)

for station, group in test_df.groupby("Station"):

    print(f"\n{station}")

    print("Records:", len(group))
    print("MAE:", round(
        mean_absolute_error(group["actual"], group["predicted"]), 4
    ))

    print("R2:", round(
        r2_score(group["actual"], group["predicted"]), 4
    ))


print("\n" + "=" * 70)
print("STATION RESULTS WITHOUT EXTREME ERRORS (> 5m)")
print("=" * 70)

clean_test = test_df[test_df["error"] <= 5].copy()

for station, group in clean_test.groupby("Station"):

    if len(group) < 2:
        continue

    print(f"\n{station}")

    print("Records:", len(group))
    print("MAE:", round(
        mean_absolute_error(group["actual"], group["predicted"]), 4
    ))

    print("R2:", round(
        r2_score(group["actual"], group["predicted"]), 4
    ))


print("\n" + "=" * 70)
print("EXTREME ERRORS (> 5 METERS)")
print("=" * 70)

print(
    test_df[
        test_df["error"] > 5
    ][
        [
            "Station",
            "timestamp",
            "river_level",
            "future_river_level",
            "actual",
            "predicted",
            "error"
        ]
    ]
    .sort_values("error", ascending=False)
    .to_string(index=False)
)

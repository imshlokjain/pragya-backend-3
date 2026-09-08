import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

DATA_FILE = "ml/data/assam/processed/assam_forecast_dataset.csv"
MODEL_FILE = "ml/models/assam_river_model.pkl"

TARGET = "future_river_level"

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print("Records:", len(df))

# Load saved model package
model_data = joblib.load(MODEL_FILE)

model = model_data["model"]
FEATURES = model_data["features"]

print("\nFeatures used by model:")
for feature in FEATURES:
    print("-", feature)

X = df[FEATURES]
y = df[TARGET]

# Must use the same split as training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Keep matching metadata
test_df = df.loc[X_test.index].copy()

print("\nLoading predictions...")
predictions = model.predict(X_test)

test_df["actual"] = y_test
test_df["predicted"] = predictions
test_df["absolute_error"] = (
    test_df["actual"] - test_df["predicted"]
).abs()

print("\n" + "=" * 70)
print("STATION-WISE PREDICTION ANALYSIS")
print("=" * 70)

for station, group in test_df.groupby("Station"):

    print(f"\n{station}")
    print("-" * 70)

    print("Records:", len(group))

    print("\nACTUAL:")
    print("Mean:", round(group["actual"].mean(), 4))
    print("Min:", round(group["actual"].min(), 4))
    print("Max:", round(group["actual"].max(), 4))

    print("\nPREDICTED:")
    print("Mean:", round(group["predicted"].mean(), 4))
    print("Min:", round(group["predicted"].min(), 4))
    print("Max:", round(group["predicted"].max(), 4))

    print("\nERROR:")
    print("MAE:", round(group["absolute_error"].mean(), 4))

    print("\nTOP 10 LARGEST ERRORS:")

    print(
        group[
            [
                "timestamp",
                "river_level",
                "rainfall_mm",
                "actual",
                "predicted",
                "absolute_error"
            ]
        ]
        .sort_values("absolute_error", ascending=False)
        .head(10)
        .to_string(index=False)
    )

print("\n" + "=" * 70)
print("OVERALL TOP 20 LARGEST ERRORS")
print("=" * 70)

print(
    test_df[
        [
            "Station",
            "timestamp",
            "river_level",
            "rainfall_mm",
            "actual",
            "predicted",
            "absolute_error"
        ]
    ]
    .sort_values("absolute_error", ascending=False)
    .head(20)
    .to_string(index=False)
)

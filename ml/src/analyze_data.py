import pandas as pd

DATA_PATH = "ml/data/raw/flood_risk_india.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DATASET INFO")
print("=" * 60)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

print(df["Flood Occurred"].value_counts())

print("\nPercentage:")
print(
    df["Flood Occurred"]
    .value_counts(normalize=True) * 100
)


numerical_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

numerical_columns.remove("Flood Occurred")


print("\n" + "=" * 60)
print("MEAN VALUES BY FLOOD OCCURRENCE")
print("=" * 60)

print(
    df.groupby("Flood Occurred")[numerical_columns]
    .mean()
    .T
)


print("\n" + "=" * 60)
print("CORRELATION WITH FLOOD OCCURRED")
print("=" * 60)

correlation = (
    df[numerical_columns + ["Flood Occurred"]]
    .corr()["Flood Occurred"]
    .sort_values(ascending=False)
)

print(correlation)


print("\n" + "=" * 60)
print("LAND COVER VS FLOOD")
print("=" * 60)

print(
    pd.crosstab(
        df["Land Cover"],
        df["Flood Occurred"],
        normalize="index",
    )
)


print("\n" + "=" * 60)
print("SOIL TYPE VS FLOOD")
print("=" * 60)

print(
    pd.crosstab(
        df["Soil Type"],
        df["Flood Occurred"],
        normalize="index",
    )
)

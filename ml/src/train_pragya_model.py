import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from xgboost import XGBClassifier


DATA_FILE = "ml/data/pragya_training_data.csv"
MODEL_FILE = "ml/models/pragya_flood_risk_xgboost.joblib"

FEATURE_COLUMNS = [
    "rainfall_mm",
    "rainfall_24h",
    "river_level",
    "river_level_change",
    "distance_to_danger",
    "historical_flood_frequency",
    "flood_affected_area",
    "flood_confidence",
]

TARGET_COLUMN = "flood_occurred"


print("=" * 60)
print("LOADING PRAGYA TRAINING DATA")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

print(f"\nDataset shape: {df.shape}")
print("\nFeatures:")
print(FEATURE_COLUMNS)

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]


print("\nTarget distribution:")
print(y.value_counts())


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


print("\nTraining XGBoost model...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
)

model.fit(X_train, y_train)


print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)
auc = roc_auc_score(y_test, probabilities)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"ROC-AUC:  {auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))


print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

importance_df = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": model.feature_importances_,
}).sort_values(
    "importance",
    ascending=False,
)

print(importance_df.to_string(index=False))


os.makedirs("ml/models", exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": FEATURE_COLUMNS,
    },
    MODEL_FILE,
)

print(f"\nModel saved successfully:")
print(MODEL_FILE)

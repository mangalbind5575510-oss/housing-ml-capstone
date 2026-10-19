import sys, os, json, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

from common.preprocessing import HousingPreprocessor

RANDOM_STATE = 42
RAW_FEATURES = ["Area_sqft", "Bedrooms", "Bathrooms", "Age_years", "Location", "Furnishing", "Parking"]

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
MODELS_DIR = os.path.join(ROOT, "models")
OUTPUT_DIR = os.path.join(ROOT, "output")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading raw dataset...")
df = pd.read_csv(os.path.join(HERE, "raw_housing_data.csv"))
print("Raw shape:", df.shape)

rows_before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Dropped {rows_before - len(df)} duplicate rows -> {df.shape}")

df["Price_Category"] = (df["Price"] > df["Price"].median()).astype(int)

X = df[RAW_FEATURES]
y_price = df["Price"]
y_cat = df["Price_Category"]

X_train, X_test, yp_train, yp_test, yc_train, yc_test = train_test_split(
    X, y_price, y_cat, test_size=0.2, random_state=RANDOM_STATE, stratify=y_cat
)
print(f"Train rows: {len(X_train)}  Test rows: {len(X_test)}")

price_pipeline = Pipeline([
    ("preprocess", HousingPreprocessor()),
    ("model", LinearRegression()),
])
price_pipeline.fit(X_train, yp_train)
yp_pred = price_pipeline.predict(X_test)

price_metrics = {
    "MAE": round(mean_absolute_error(yp_test, yp_pred), 2),
    "RMSE": round(float(np.sqrt(mean_squared_error(yp_test, yp_pred))), 2),
    "R2": round(r2_score(yp_test, yp_pred), 4),
}
print("Price model metrics:", price_metrics)

category_pipeline = Pipeline([
    ("preprocess", HousingPreprocessor()),
    ("model", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_STATE)),
])
category_pipeline.fit(X_train, yc_train)
yc_pred = category_pipeline.predict(X_test)

cat_metrics = {
    "Accuracy": round(accuracy_score(yc_test, yc_pred), 4),
    "Precision": round(precision_score(yc_test, yc_pred), 4),
    "Recall": round(recall_score(yc_test, yc_pred), 4),
    "F1": round(f1_score(yc_test, yc_pred), 4),
    "ConfusionMatrix": confusion_matrix(yc_test, yc_pred).tolist(),
}
print("Category model metrics:", cat_metrics)

price_path = os.path.join(MODELS_DIR, "price_pipeline.joblib")
cat_path = os.path.join(MODELS_DIR, "category_pipeline.joblib")
joblib.dump(price_pipeline, price_path)
joblib.dump(category_pipeline, cat_path)
print(f"Saved: {price_path}")
print(f"Saved: {cat_path}")

metadata = {
    "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "random_state": RANDOM_STATE,
    "raw_input_features": RAW_FEATURES,
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "price_model": {
        "algorithm": "LinearRegression",
        "target": "Price (continuous, USD)",
        "metrics": price_metrics,
        "file": "models/price_pipeline.joblib",
    },
    "category_model": {
        "algorithm": "RandomForestClassifier (n_estimators=200, max_depth=6)",
        "target": "Price_Category (0=Low, 1=High; split at training median)",
        "price_median_threshold": round(float(df["Price"].median()), 2),
        "metrics": cat_metrics,
        "file": "models/category_pipeline.joblib",
    },
}
with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
print("Saved: output/metadata.json")
print("\nTraining & serialization complete.")

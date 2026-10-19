import sys, os, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from common.preprocessing import HousingPreprocessor  # noqa: F401

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_FEATURES = ["Area_sqft", "Bedrooms", "Bathrooms", "Age_years", "Location", "Furnishing", "Parking"]
RANDOM_STATE = 42

df = pd.read_csv(os.path.join(ROOT, "training", "raw_housing_data.csv")).drop_duplicates().reset_index(drop=True)
df["Price_Category"] = (df["Price"] > df["Price"].median()).astype(int)

X = df[RAW_FEATURES]
y_price = df["Price"]
y_cat = df["Price_Category"]
X_train, X_test, yp_train, yp_test, yc_train, yc_test = train_test_split(
    X, y_price, y_cat, test_size=0.2, random_state=RANDOM_STATE, stratify=y_cat
)

price_pipeline = joblib.load(os.path.join(ROOT, "models", "price_pipeline.joblib"))
category_pipeline = joblib.load(os.path.join(ROOT, "models", "category_pipeline.joblib"))

yp_pred = price_pipeline.predict(X_test)
yc_pred = category_pipeline.predict(X_test)

os.makedirs(os.path.join(ROOT, "plots"), exist_ok=True)

# Predicted vs Actual
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(yp_test, yp_pred, alpha=0.5, s=20, color="#4C72B0")
lims = [min(yp_test.min(), yp_pred.min()), max(yp_test.max(), yp_pred.max())]
ax.plot(lims, lims, "r--", linewidth=1)
ax.set_xlabel("Actual Price ($)")
ax.set_ylabel("Predicted Price ($)")
ax.set_title("Deployed Price Pipeline: Predicted vs Actual (test set)")
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "plots", "price_predicted_vs_actual.png"), dpi=130)
plt.close()

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yc_test, yc_pred)
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Low", "High"]); ax.set_yticklabels(["Low", "High"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Deployed Category Pipeline - Confusion Matrix")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "plots", "category_confusion_matrix.png"), dpi=130)
plt.close()

print("Report plots saved to plots/")

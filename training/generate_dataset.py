import os
import numpy as np
import pandas as pd

np.random.seed(42)
n = 500
HERE = os.path.dirname(__file__)

locations = ["Downtown", "Suburb", "Rural", "Uptown", "Waterfront"]
furnishing = ["Unfurnished", "Semi-Furnished", "Fully-Furnished"]
parking = ["Yes", "No"]

data = {
    "Id": np.arange(1, n + 1),
    "Area_sqft": np.random.normal(1800, 600, n).round(0),
    "Bedrooms": np.random.randint(1, 6, n),
    "Bathrooms": np.random.randint(1, 4, n).astype(float),
    "Location": np.random.choice(locations, n, p=[0.25, 0.3, 0.15, 0.2, 0.1]),
    "Furnishing": np.random.choice(furnishing, n, p=[0.4, 0.35, 0.25]),
    "Age_years": np.random.randint(0, 40, n).astype(float),
    "Parking": np.random.choice(parking, n, p=[0.7, 0.3]),
}
df = pd.DataFrame(data)

loc_premium = df["Location"].map({"Downtown": 60, "Waterfront": 70, "Uptown": 40, "Suburb": 20, "Rural": 0})
furnish_premium = df["Furnishing"].map({"Fully-Furnished": 25, "Semi-Furnished": 12, "Unfurnished": 0})
df["Price"] = (
    df["Area_sqft"] * 120 + df["Bedrooms"] * 15000 + df["Bathrooms"] * 8000
    - df["Age_years"] * 900 + loc_premium * 1000 + furnish_premium * 1000
    + np.random.normal(0, 20000, n)
).round(0)

outlier_idx = np.random.choice(n, 6, replace=False)
df.loc[outlier_idx[:3], "Area_sqft"] = df["Area_sqft"].max() * np.random.uniform(3, 4, 3)
df.loc[outlier_idx[3:], "Price"] = df["Price"].max() * np.random.uniform(3, 5, 3)

for col, frac in [("Area_sqft", 0.06), ("Bathrooms", 0.05), ("Location", 0.04),
                   ("Furnishing", 0.05), ("Age_years", 0.07)]:
    idx = np.random.choice(n, int(n * frac), replace=False)
    df.loc[idx, col] = np.nan

dup_rows = df.sample(4, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)

df.to_csv(os.path.join(HERE, "raw_housing_data.csv"), index=False)
print("Generated raw_housing_data.csv:", df.shape)

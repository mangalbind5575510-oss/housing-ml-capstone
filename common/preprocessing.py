"""
Shared preprocessing transformer for the Housing Price / Category capstone project.
"""
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class HousingPreprocessor(BaseEstimator, TransformerMixin):
    NUMERIC_COLS = ["Area_sqft", "Bedrooms", "Bathrooms", "Age_years"]
    MODE_FILL_COLS = ["Location", "Furnishing"]
    FURNISHING_ORDER = {"Unfurnished": 0, "Semi-Furnished": 1, "Fully-Furnished": 2}
    PARKING_MAP = {"No": 0, "Yes": 1}

    def fit(self, X, y=None):
        X = X.copy()
        self.medians_ = {c: float(X[c].median()) for c in self.NUMERIC_COLS}
        self.modes_ = {c: X[c].mode()[0] for c in self.MODE_FILL_COLS}
        self.parking_mode_ = X["Parking"].mode()[0] if "Parking" in X.columns else "No"

        X_filled = X.copy()
        for c in self.NUMERIC_COLS:
            X_filled[c] = X_filled[c].fillna(self.medians_[c])
        for c in self.MODE_FILL_COLS:
            X_filled[c] = X_filled[c].fillna(self.modes_[c])

        q1, q3 = X_filled["Area_sqft"].quantile([0.25, 0.75])
        iqr = q3 - q1
        self.area_bounds_ = (float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr))
        X_filled["Area_sqft"] = X_filled["Area_sqft"].clip(*self.area_bounds_)

        self.location_categories_ = sorted(X_filled["Location"].unique().tolist())

        self.scale_min_ = {c: float(X_filled[c].min()) for c in self.NUMERIC_COLS}
        self.scale_max_ = {c: float(X_filled[c].max()) for c in self.NUMERIC_COLS}

        self.feature_names_out_ = (
            self.NUMERIC_COLS
            + ["Furnishing_encoded", "Parking_encoded"]
            + [f"Location_{c}" for c in self.location_categories_[1:]]
        )
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.NUMERIC_COLS:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(self.medians_[c])
        for c in self.MODE_FILL_COLS:
            X[c] = X[c].fillna(self.modes_[c])
        if "Parking" in X.columns:
            X["Parking"] = X["Parking"].fillna(self.parking_mode_)
        else:
            X["Parking"] = self.parking_mode_

        X["Area_sqft"] = X["Area_sqft"].clip(*self.area_bounds_)

        X["Furnishing_encoded"] = X["Furnishing"].map(self.FURNISHING_ORDER)
        X["Furnishing_encoded"] = X["Furnishing_encoded"].fillna(
            self.FURNISHING_ORDER[self.modes_["Furnishing"]]
        ).astype(int)

        X["Parking_encoded"] = X["Parking"].map(self.PARKING_MAP)
        X["Parking_encoded"] = X["Parking_encoded"].fillna(0).astype(int)

        for cat in self.location_categories_[1:]:
            X[f"Location_{cat}"] = (X["Location"] == cat).astype(int)

        for c in self.NUMERIC_COLS:
            lo, hi = self.scale_min_[c], self.scale_max_[c]
            denom = (hi - lo) if hi > lo else 1.0
            X[c] = (X[c] - lo) / denom

        return X[self.feature_names_out_].astype(float).values

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)

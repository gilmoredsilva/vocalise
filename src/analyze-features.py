import pandas as pd
import joblib
import numpy as np


# Load feature dataset
df = pd.read_csv("dataset/features.csv")

# Metadata columns
metadata_columns = ["file", "actor", "emotion"]

feature_columns = [
    column for column in df.columns
    if column not in metadata_columns
]


# Load best model and scaler
model = joblib.load(
    "models/logistic_regression_balanced.pkl"
)

scaler = joblib.load(
    "models/logistic_regression_balanced_scaler.pkl"
)


# Logistic Regression coefficients
coefficients = model.coef_

# Mean absolute coefficient across all 6 classes
importance = np.mean(
    np.abs(coefficients),
    axis=0
)


# Create feature importance table
importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": importance
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)


# Display all features
print("\nFeature Importance Ranking")
print("==========================")

print(importance_df.to_string(index=False))


# Save results
importance_df.to_csv(
    "results/feature_importance.csv",
    index=False
)

print("\nSaved to:")
print("results/feature_importance.csv")
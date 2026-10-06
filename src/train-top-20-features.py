import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load feature dataset
df = pd.read_csv("dataset/features.csv")

# Load speaker-independent splits
train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")

train_files = set(train_df["file"])
test_files = set(test_df["file"])

train_data = df[df["file"].isin(train_files)]
test_data = df[df["file"].isin(test_files)]


# --------------------------------------------------
# Top 20 features from our feature-importance analysis
# --------------------------------------------------

top_features = [
    "energy_mean",
    "spectral_centroid_mean",
    "spectral_bandwidth_mean",
    "spectral_rolloff_mean",
    "zcr_mean",
    "energy_std",
    "mfcc_1_mean",
    "mfcc_2_mean",
    "mfcc_1_std",
    "spectral_flux_mean",
    "mfcc_3_std",
    "mfcc_3_mean",
    "spectral_centroid_std",
    "spectral_contrast_6_mean",
    "mfcc_4_std",
    "spectral_rolloff_std",
    "mfcc_5_std",
    "mfcc_5_mean",
    "mfcc_2_std",
    "pitch_mean"
]


# Verify features exist
missing_features = [
    feature for feature in top_features
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing features: {missing_features}"
    )


X_train = train_data[top_features]
y_train = train_data["emotion"]

X_test = test_data[top_features]
y_test = test_data["emotion"]


# --------------------------------------------------
# Standardization
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# Balanced Logistic Regression
# --------------------------------------------------

model = LogisticRegression(
    C=100,
    class_weight="balanced",
    max_iter=2000,
    random_state=42
)

print("Training Top-20 Feature Logistic Regression...")
model.fit(X_train_scaled, y_train)

print("Training complete!")


# --------------------------------------------------
# Predictions
# --------------------------------------------------

y_pred = model.predict(X_test_scaled)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# --------------------------------------------------
# Save model and scaler
# --------------------------------------------------

joblib.dump(
    model,
    "models/logistic_regression_top20.pkl"
)

joblib.dump(
    scaler,
    "models/logistic_regression_top20_scaler.pkl"
)

print("\nModel saved to:")
print("models/logistic_regression_top20.pkl")

print("Scaler saved to:")
print("models/logistic_regression_top20_scaler.pkl")
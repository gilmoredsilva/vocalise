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


# Separate metadata from numerical features
metadata_columns = ["file", "actor", "emotion"]

feature_columns = [
    column for column in df.columns
    if column not in metadata_columns
]

X_train = train_data[feature_columns]
y_train = train_data["emotion"]

X_test = test_data[feature_columns]
y_test = test_data["emotion"]


# Standardize features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train balanced Logistic Regression
model = LogisticRegression(
    C=100,
    class_weight="balanced",
    max_iter=2000,
    random_state=42
)

print("Training balanced Logistic Regression...")
model.fit(X_train_scaled, y_train)

print("Training complete!")


# Predictions
y_pred = model.predict(X_test_scaled)


# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Save model and scaler
joblib.dump(
    model,
    "models/logistic_regression_balanced.pkl"
)

joblib.dump(
    scaler,
    "models/logistic_regression_balanced_scaler.pkl"
)

print("\nModel saved to:")
print("models/logistic_regression_balanced.pkl")

print("Scaler saved to:")
print("models/logistic_regression_balanced_scaler.pkl")
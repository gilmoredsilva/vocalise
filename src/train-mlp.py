import pandas as pd
import joblib

from sklearn.neural_network import MLPClassifier
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


# -------------------------
# Standardize features
# -------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -------------------------
# Train MLP
# -------------------------

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    learning_rate_init=0.001,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.15,
    random_state=42
)

print("Training MLP...")
model.fit(X_train_scaled, y_train)

print("Training complete!")


# -------------------------
# Predictions
# -------------------------

y_pred = model.predict(X_test_scaled)


# -------------------------
# Evaluation
# -------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# -------------------------
# Save model and scaler
# -------------------------

joblib.dump(model, "models/mlp_model.pkl")
joblib.dump(scaler, "models/mlp_scaler.pkl")

print("\nModel saved to:")
print("models/mlp_model.pkl")

print("Scaler saved to:")
print("models/mlp_scaler.pkl")
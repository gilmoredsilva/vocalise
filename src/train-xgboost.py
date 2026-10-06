import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
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


# Encode emotion labels
label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


# -------------------------
# Train XGBoost
# -------------------------

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=6,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

print("Training XGBoost...")
model.fit(X_train, y_train_encoded)

print("Training complete!")


# -------------------------
# Predictions
# -------------------------

y_pred_encoded = model.predict(X_test)

y_pred = label_encoder.inverse_transform(
    y_pred_encoded.astype(int)
)


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
# Save model
# -------------------------

joblib.dump(
    model,
    "models/xgboost_model.pkl"
)

joblib.dump(
    label_encoder,
    "models/xgboost_label_encoder.pkl"
)

print("\nModel saved to:")
print("models/xgboost_model.pkl")

print("Label encoder saved to:")
print("models/xgboost_label_encoder.pkl")
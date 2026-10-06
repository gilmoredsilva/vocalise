import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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
# Hyperparameter tuning
# -------------------------

parameter_grid = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=parameter_grid,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=1
)

print("Tuning Logistic Regression...")
grid_search.fit(X_train_scaled, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest CV Macro F1:")
print(grid_search.best_score_)


# -------------------------
# Evaluate best model
# -------------------------

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# -------------------------
# Save model and scaler
# -------------------------

joblib.dump(
    best_model,
    "models/logistic_regression_tuned.pkl"
)

joblib.dump(
    scaler,
    "models/logistic_regression_tuned_scaler.pkl"
)

print("\nModel saved to:")
print("models/logistic_regression_tuned.pkl")

print("Scaler saved to:")
print("models/logistic_regression_tuned_scaler.pkl")
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load extracted features
df = pd.read_csv("dataset/features.csv")

# Load our speaker-independent split information
train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")

# Use file paths to identify train/test samples
train_files = set(train_df["file"])
test_files = set(test_df["file"])

train_data = df[df["file"].isin(train_files)].copy()
test_data = df[df["file"].isin(test_files)].copy()

# Columns that are NOT ML features
metadata_columns = ["file", "actor", "emotion"]

feature_columns = [
    column for column in df.columns
    if column not in metadata_columns
]

X_train = train_data[feature_columns]
y_train = train_data["emotion"]

X_test = test_data[feature_columns]
y_test = test_data["emotion"]

# Standardize features using ONLY training data
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

print("Number of features:", X_train.shape[1])

print("\nTraining classes:")
print(y_train.value_counts())

print("\nTesting classes:")
print(y_test.value_counts())

print("\nScaled training shape:", X_train_scaled.shape)
print("Scaled testing shape:", X_test_scaled.shape)
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

df = pd.read_csv("dataset/dataset.csv")

# First split: 80% actors for training, 20% actors for testing
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(df, groups=df["actor"])
)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

# Save the splits
train_df.to_csv("dataset/train.csv", index=False)
test_df.to_csv("dataset/test.csv", index=False)

print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))

print("\nTraining actors:", train_df["actor"].nunique())
print("Testing actors:", test_df["actor"].nunique())

print("\nTraining emotions:")
print(train_df["emotion"].value_counts())

print("\nTesting emotions:")
print(test_df["emotion"].value_counts())

# Verify there is no actor overlap
overlap = set(train_df["actor"]) & set(test_df["actor"])

print("\nActor overlap:", overlap)
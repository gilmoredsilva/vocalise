import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv("dataset/train.csv")
features_df = pd.read_csv("dataset/features.csv")


# ============================================================
# FINAL 54 FEATURES
# ============================================================

feature_columns = [
    "mfcc_1_mean", "mfcc_2_mean", "mfcc_3_mean", "mfcc_4_mean",
    "mfcc_5_mean", "mfcc_6_mean", "mfcc_7_mean", "mfcc_8_mean",
    "mfcc_9_mean", "mfcc_10_mean", "mfcc_11_mean", "mfcc_12_mean",
    "mfcc_13_mean",

    "mfcc_1_std", "mfcc_2_std", "mfcc_3_std", "mfcc_4_std",
    "mfcc_5_std", "mfcc_6_std", "mfcc_7_std", "mfcc_8_std",
    "mfcc_9_std", "mfcc_10_std", "mfcc_11_std", "mfcc_12_std",
    "mfcc_13_std",

    "zcr_mean", "zcr_std",

    "energy_mean", "energy_std",

    "spectral_centroid_mean",
    "spectral_centroid_std",

    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",

    "spectral_rolloff_mean",
    "spectral_rolloff_std",

    "pitch_mean",
    "pitch_std",

    "spectral_contrast_1_mean",
    "spectral_contrast_2_mean",
    "spectral_contrast_3_mean",
    "spectral_contrast_4_mean",
    "spectral_contrast_5_mean",
    "spectral_contrast_6_mean",
    "spectral_contrast_7_mean",

    "spectral_contrast_1_std",
    "spectral_contrast_2_std",
    "spectral_contrast_3_std",
    "spectral_contrast_4_std",
    "spectral_contrast_5_std",
    "spectral_contrast_6_std",
    "spectral_contrast_7_std",

    "spectral_flux_mean",
    "spectral_flux_std"
]


# ============================================================
# MATCH TRAINING DATA
# ============================================================

train_features = features_df[
    features_df["file"].isin(train_df["file"])
].copy()

train_features = (
    train_features
    .set_index("file")
    .loc[train_df["file"]]
    .reset_index()
)


X = train_features[feature_columns]
y = train_features["emotion"]
groups = train_features["actor"]


print("Training samples:", len(X))
print("Features:", len(feature_columns))
print("Unique speakers:", groups.nunique())


# ============================================================
# MODEL PIPELINE
# ============================================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),

    ("classifier", LogisticRegression(
        C=100,
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    ))
])


# ============================================================
# SPEAKER-INDEPENDENT CROSS VALIDATION
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

accuracies = []
macro_f1_scores = []


print("\nRunning 5-fold speaker-independent cross-validation...\n")


for fold, (train_idx, val_idx) in enumerate(
    cv.split(X, y, groups),
    start=1
):

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)

    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro"
    )

    accuracies.append(accuracy)
    macro_f1_scores.append(macro_f1)

    print(
        f"Fold {fold}: "
        f"Accuracy = {accuracy:.4f}, "
        f"Macro F1 = {macro_f1:.4f}"
    )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 55)
print("CROSS-VALIDATION RESULTS")
print("=" * 55)

print(
    f"\nMean Accuracy: "
    f"{np.mean(accuracies):.4f}"
)

print(
    f"Std Accuracy: "
    f"{np.std(accuracies):.4f}"
)

print(
    f"\nMean Macro F1: "
    f"{np.mean(macro_f1_scores):.4f}"
)

print(
    f"Std Macro F1: "
    f"{np.std(macro_f1_scores):.4f}"
)

print("\nIndividual fold accuracies:")
for i, score in enumerate(accuracies, 1):
    print(f"Fold {i}: {score:.4f}")

print("\nIndividual fold Macro F1:")
for i, score in enumerate(macro_f1_scores, 1):
    print(f"Fold {i}: {score:.4f}")
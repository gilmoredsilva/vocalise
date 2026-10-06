import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")
features_df = pd.read_csv("dataset/features.csv")


# ============================================================
# 54 FINAL FEATURES
# ============================================================

feature_columns = [
    # MFCC means
    "mfcc_1_mean",
    "mfcc_2_mean",
    "mfcc_3_mean",
    "mfcc_4_mean",
    "mfcc_5_mean",
    "mfcc_6_mean",
    "mfcc_7_mean",
    "mfcc_8_mean",
    "mfcc_9_mean",
    "mfcc_10_mean",
    "mfcc_11_mean",
    "mfcc_12_mean",
    "mfcc_13_mean",

    # MFCC standard deviations
    "mfcc_1_std",
    "mfcc_2_std",
    "mfcc_3_std",
    "mfcc_4_std",
    "mfcc_5_std",
    "mfcc_6_std",
    "mfcc_7_std",
    "mfcc_8_std",
    "mfcc_9_std",
    "mfcc_10_std",
    "mfcc_11_std",
    "mfcc_12_std",
    "mfcc_13_std",

    # ZCR
    "zcr_mean",
    "zcr_std",

    # Energy
    "energy_mean",
    "energy_std",

    # Spectral centroid
    "spectral_centroid_mean",
    "spectral_centroid_std",

    # Spectral bandwidth
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",

    # Spectral rolloff
    "spectral_rolloff_mean",
    "spectral_rolloff_std",

    # Pitch
    "pitch_mean",
    "pitch_std",

    # Spectral contrast
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

    # Spectral flux
    "spectral_flux_mean",
    "spectral_flux_std"
]


# ============================================================
# MATCH TRAIN / TEST FEATURES
# ============================================================

train_features = features_df[
    features_df["file"].isin(train_df["file"])
].copy()

test_features = features_df[
    features_df["file"].isin(test_df["file"])
].copy()


train_features = (
    train_features
    .set_index("file")
    .loc[train_df["file"]]
    .reset_index()
)

test_features = (
    test_features
    .set_index("file")
    .loc[test_df["file"]]
    .reset_index()
)


# ============================================================
# X / Y
# ============================================================

X_train = train_features[feature_columns]
X_test = test_features[feature_columns]

y_train = train_features["emotion"]
y_test = test_features["emotion"]


print("Training samples:", len(X_train))
print("Test samples:", len(X_test))
print("Features:", len(feature_columns))


# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# BALANCED RBF SVM
# ============================================================

print("\nTraining Balanced RBF SVM...")

model = SVC(
    C=10,
    gamma=0.01,
    kernel="rbf",
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("Training complete!")


# ============================================================
# EVALUATION
# ============================================================

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "models/svm_balanced.pkl"
)

joblib.dump(
    scaler,
    "models/svm_balanced_scaler.pkl"
)

print("\nModel saved to:")
print("models/svm_balanced.pkl")

print("\nScaler saved to:")
print("models/svm_balanced_scaler.pkl")
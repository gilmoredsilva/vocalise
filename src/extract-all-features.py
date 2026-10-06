from pathlib import Path

import librosa
import numpy as np
import pandas as pd


FRAME_LENGTH_MS = 25
HOP_LENGTH_MS = 15
N_MFCC = 13
PRE_EMPHASIS = 0.97


def extract_features(audio_file):
    # Load audio
    signal, sample_rate = librosa.load(audio_file, sr=None)

    # -------------------------
    # Pre-emphasis
    # -------------------------

    signal = np.append(
        signal[0],
        signal[1:] - PRE_EMPHASIS * signal[:-1]
    )

    # -------------------------
    # Frame parameters
    # -------------------------

    frame_length = int(
        FRAME_LENGTH_MS / 1000 * sample_rate
    )

    hop_length = int(
        HOP_LENGTH_MS / 1000 * sample_rate
    )

    # -------------------------
    # MFCC
    # -------------------------

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=N_MFCC,
        n_fft=frame_length,
        hop_length=hop_length
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # -------------------------
    # Zero Crossing Rate
    # -------------------------

    zcr = librosa.feature.zero_crossing_rate(
        signal,
        frame_length=frame_length,
        hop_length=hop_length
    )

    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    # -------------------------
    # Short-Time Energy / RMS
    # -------------------------

    energy = librosa.feature.rms(
        y=signal,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    energy_mean = np.mean(energy)
    energy_std = np.std(energy)

    # -------------------------
    # Spectral Centroid
    # -------------------------

    centroid = librosa.feature.spectral_centroid(
        y=signal,
        sr=sample_rate,
        n_fft=frame_length,
        hop_length=hop_length
    )[0]

    centroid_mean = np.mean(centroid)
    centroid_std = np.std(centroid)

    # -------------------------
    # Spectral Bandwidth
    # -------------------------

    bandwidth = librosa.feature.spectral_bandwidth(
        y=signal,
        sr=sample_rate,
        n_fft=frame_length,
        hop_length=hop_length
    )[0]

    bandwidth_mean = np.mean(bandwidth)
    bandwidth_std = np.std(bandwidth)

    # -------------------------
    # Spectral Rolloff
    # -------------------------

    rolloff = librosa.feature.spectral_rolloff(
        y=signal,
        sr=sample_rate,
        n_fft=frame_length,
        hop_length=hop_length
    )[0]

    rolloff_mean = np.mean(rolloff)
    rolloff_std = np.std(rolloff)

    # -------------------------
    # Pitch / Fundamental Frequency
    # -------------------------

    pitch = librosa.yin(
        signal,
        fmin=65,
        fmax=500,
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length
    )

    pitch = pitch[np.isfinite(pitch)]

    if len(pitch) == 0:
        pitch_mean = 0.0
        pitch_std = 0.0
    else:
        pitch_mean = np.mean(pitch)
        pitch_std = np.std(pitch)

    # -------------------------
    # Spectral Contrast
    # -------------------------

    contrast = librosa.feature.spectral_contrast(
        y=signal,
        sr=sample_rate,
        n_fft=frame_length,
        hop_length=hop_length
    )

    contrast_mean = np.mean(contrast, axis=1)
    contrast_std = np.std(contrast, axis=1)

    # -------------------------
    # Spectral Flux
    # -------------------------

    stft = np.abs(
        librosa.stft(
            signal,
            n_fft=frame_length,
            hop_length=hop_length
        )
    )

    stft_norm = stft / (
        np.sum(stft, axis=0, keepdims=True) + 1e-10
    )

    spectral_flux = np.sqrt(
        np.sum(
            np.diff(stft_norm, axis=1) ** 2,
            axis=0
        )
    )

    spectral_flux_mean = np.mean(spectral_flux)
    spectral_flux_std = np.std(spectral_flux)

    # -------------------------
    # Combine features
    # -------------------------

    features = np.concatenate([
        # MFCC
        mfcc_mean,
        mfcc_std,

        # Existing features
        [zcr_mean, zcr_std],
        [energy_mean, energy_std],
        [centroid_mean, centroid_std],
        [bandwidth_mean, bandwidth_std],
        [rolloff_mean, rolloff_std],
        [pitch_mean, pitch_std],

        # Spectral Contrast
        contrast_mean,
        contrast_std,

        # Spectral Flux
        [spectral_flux_mean, spectral_flux_std]
    ])

    return features


# --------------------------------
# Load dataset
# --------------------------------

df = pd.read_csv("dataset/dataset.csv")

feature_rows = []

print("Starting feature extraction...")
print("Total files:", len(df))

for i, row in df.iterrows():

    features = extract_features(row["file"])

    feature_rows.append(features)

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(df)} files")


# --------------------------------
# Create feature DataFrame
# --------------------------------

feature_names = []

# MFCC means
for i in range(1, 14):
    feature_names.append(f"mfcc_{i}_mean")

# MFCC standard deviations
for i in range(1, 14):
    feature_names.append(f"mfcc_{i}_std")

# Existing features
feature_names += [
    "zcr_mean",
    "zcr_std",
    "energy_mean",
    "energy_std",
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",
    "spectral_rolloff_mean",
    "spectral_rolloff_std",
    "pitch_mean",
    "pitch_std"
]

# Spectral Contrast
for i in range(1, 8):
    feature_names.append(f"spectral_contrast_{i}_mean")

for i in range(1, 8):
    feature_names.append(f"spectral_contrast_{i}_std")

# Spectral Flux
feature_names += [
    "spectral_flux_mean",
    "spectral_flux_std"
]


features_df = pd.DataFrame(
    feature_rows,
    columns=feature_names
)

# Add metadata
features_df.insert(0, "file", df["file"].values)
features_df.insert(1, "actor", df["actor"].values)
features_df.insert(2, "emotion", df["emotion"].values)

# Save
features_df.to_csv(
    "dataset/features.csv",
    index=False
)

print("\nFeature extraction complete!")
print("Feature matrix shape:", features_df.shape)
print("Total ML features:", len(feature_names))
print("Saved to: dataset/features.csv")
import librosa
import pandas as pd
import numpy as np

df = pd.read_csv("dataset/train.csv")

audio_file = df.iloc[0]["file"]

signal, sample_rate = librosa.load(audio_file, sr=None)

frame_length = int(0.025 * sample_rate)
hop_length = int(0.015 * sample_rate)

# -------------------------
# MFCC
# -------------------------

mfcc = librosa.feature.mfcc(
    y=signal,
    sr=sample_rate,
    n_mfcc=13,
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
# Short-Time Energy
# -------------------------

frames = librosa.util.frame(
    signal,
    frame_length=frame_length,
    hop_length=hop_length
)

energy = np.mean(frames ** 2, axis=0)

energy_mean = np.mean(energy)
energy_std = np.std(energy)

# -------------------------
# Spectral Centroid
# -------------------------

spectral_centroid = librosa.feature.spectral_centroid(
    y=signal,
    sr=sample_rate,
    n_fft=frame_length,
    hop_length=hop_length
)

centroid_mean = np.mean(spectral_centroid)
centroid_std = np.std(spectral_centroid)

# -------------------------
# Combine everything
# -------------------------

features = np.concatenate([
    mfcc_mean,
    mfcc_std,
    [zcr_mean, zcr_std],
    [energy_mean, energy_std],
    [centroid_mean, centroid_std]
])

print("Audio:", audio_file)
print("Number of features:", len(features))

print("\nFeature vector:")
print(features)
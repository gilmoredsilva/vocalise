import librosa
import pandas as pd
import numpy as np

df = pd.read_csv("dataset/train.csv")

audio_file = df.iloc[0]["file"]

signal, sample_rate = librosa.load(audio_file, sr=None)

# Same framing parameters we've been using
frame_length = int(0.025 * sample_rate)
hop_length = int(0.015 * sample_rate)

# 1. Zero Crossing Rate
zcr = librosa.feature.zero_crossing_rate(
    signal,
    frame_length=frame_length,
    hop_length=hop_length
)

# 2. Short-Time Energy
frames = librosa.util.frame(
    signal,
    frame_length=frame_length,
    hop_length=hop_length
)

energy = np.mean(frames ** 2, axis=0)

# 3. Spectral Centroid
spectral_centroid = librosa.feature.spectral_centroid(
    y=signal,
    sr=sample_rate,
    n_fft=frame_length,
    hop_length=hop_length
)

print("Audio:", audio_file)

print("\nZCR:")
print("Shape:", zcr.shape)
print("Mean:", np.mean(zcr))

print("\nShort-Time Energy:")
print("Shape:", energy.shape)
print("Mean:", np.mean(energy))

print("\nSpectral Centroid:")
print("Shape:", spectral_centroid.shape)
print("Mean:", np.mean(spectral_centroid), "Hz")
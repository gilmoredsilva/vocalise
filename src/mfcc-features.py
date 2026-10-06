import librosa
import pandas as pd
import numpy as np

df = pd.read_csv("dataset/train.csv")

audio_file = df.iloc[0]["file"]

signal, sample_rate = librosa.load(audio_file, sr=None)

# MFCC extraction
mfcc = librosa.feature.mfcc(
    y=signal,
    sr=sample_rate,
    n_mfcc=13,
    n_fft=int(0.025 * sample_rate),
    hop_length=int(0.015 * sample_rate)
)

print("Audio:", audio_file)
print("Sample rate:", sample_rate)

print("\nMFCC matrix shape:", mfcc.shape)

print("\nMean of each MFCC coefficient:")
print(np.mean(mfcc, axis=1))

print("\nStandard deviation of each MFCC coefficient:")
print(np.std(mfcc, axis=1))
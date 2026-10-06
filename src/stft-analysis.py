import librosa
import librosa.display
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("dataset/train.csv")

audio_file = df.iloc[0]["file"]

signal, sample_rate = librosa.load(audio_file, sr=None)

# STFT parameters
n_fft = int(0.025 * sample_rate)
hop_length = int(0.015 * sample_rate)

# Short-Time Fourier Transform
stft = librosa.stft(
    signal,
    n_fft=n_fft,
    hop_length=hop_length,
    window="hamming"
)

# Convert magnitude to decibels
spectrogram = librosa.amplitude_to_db(
    np.abs(stft),
    ref=np.max
)

# Plot spectrogram
plt.figure(figsize=(12, 6))

librosa.display.specshow(
    spectrogram,
    sr=sample_rate,
    hop_length=hop_length,
    x_axis="time",
    y_axis="hz"
)

plt.colorbar(format="%+2.0f dB")
plt.title("Speech Spectrogram")
plt.ylim(0, 8000)
plt.tight_layout()

plt.show()
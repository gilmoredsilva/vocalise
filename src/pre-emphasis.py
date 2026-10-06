import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("dataset/train.csv")

audio_file = df.iloc[0]["file"]

signal, sample_rate = librosa.load(audio_file, sr=None)

alpha = 0.97

# Pre-emphasis
emphasized_signal = np.append(
    signal[0],
    signal[1:] - alpha * signal[:-1]
)

# Plot original and pre-emphasized signals
plt.figure(figsize=(12, 4))
plt.plot(signal, label="Original")
plt.plot(emphasized_signal, label="Pre-emphasized", alpha=0.7)

plt.title("Speech Signal Before and After Pre-emphasis")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.show()
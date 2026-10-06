import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/train.csv")

audio_file = df.iloc[0]["file"]

signal, sample_rate = librosa.load(audio_file, sr=None)

# Frame parameters
frame_length = int(0.025 * sample_rate)
hop_length = int(0.015 * sample_rate)

# Take the first frame
frame = signal[:frame_length]

# Create Hamming window
hamming_window = np.hamming(frame_length)

# Apply window
windowed_frame = frame * hamming_window

# Plot before and after windowing
plt.figure(figsize=(12, 5))

plt.plot(frame, label="Original Frame")
plt.plot(windowed_frame, label="Hamming Windowed Frame")

plt.title("Speech Frame Before and After Hamming Window")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()

plt.show()
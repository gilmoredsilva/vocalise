import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/train.csv")

audio_file = df.iloc[0]["file"]

signal, sample_rate = librosa.load(audio_file, sr=None)

# Take one 25 ms frame
frame_length = int(0.025 * sample_rate)
frame = signal[:frame_length]

# Apply Hamming window
window = np.hamming(frame_length)
windowed_frame = frame * window

# FFT
fft_result = np.fft.rfft(windowed_frame)

# Magnitude spectrum
magnitude = np.abs(fft_result)

# Frequency axis
frequencies = np.fft.rfftfreq(
    len(windowed_frame),
    1 / sample_rate
)

# Plot frequency spectrum
plt.figure(figsize=(12, 5))

plt.plot(frequencies, magnitude)

plt.title("Frequency Spectrum of Speech Frame")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

plt.xlim(0, 8000)

plt.tight_layout()
plt.show()
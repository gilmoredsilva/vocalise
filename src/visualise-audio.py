import librosa
import matplotlib.pyplot as plt
import pandas as pd

# Load our training metadata
df = pd.read_csv("dataset/train.csv")

# Select the first audio file
audio_file = df.iloc[0]["file"]

# Load audio without changing its original sampling rate
signal, sample_rate = librosa.load(audio_file, sr=None)

# Create time axis
time = [i / sample_rate for i in range(len(signal))]

# Plot waveform
plt.figure(figsize=(12, 4))
plt.plot(time, signal)

plt.title("Speech Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.tight_layout()

plt.show()
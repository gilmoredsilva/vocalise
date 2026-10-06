import librosa
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/train.csv")

audio_file = df.iloc[0]["file"]

signal, sample_rate = librosa.load(audio_file, sr=None)

frame_duration = 0.025
hop_duration = 0.015

frame_length = int(frame_duration * sample_rate)
hop_length = int(hop_duration * sample_rate)

print("Sample rate:", sample_rate)
print("Frame length:", frame_length, "samples")
print("Hop length:", hop_length, "samples")

# Manually extract only the first 5 frames
frames = []

for i in range(5):
    start = i * hop_length
    end = start + frame_length
    frames.append(signal[start:end])

print("Extracted:", len(frames), "frames")

plt.figure(figsize=(12, 5))

for i, frame in enumerate(frames):
    plt.plot(frame + i * 0.5)

plt.title("First Five Speech Frames")
plt.xlabel("Sample within frame")
plt.ylabel("Amplitude + offset")
plt.tight_layout()
plt.show()
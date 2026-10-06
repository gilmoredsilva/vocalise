from pathlib import Path

dataset_path = Path("dataset/archive/AudioWAV")

wav_files = list(dataset_path.rglob("*.wav"))

for file in wav_files[:10]:
    parts = file.stem.split("_")

    actor = parts[0]
    sentence = parts[1]
    emotion = parts[2]
    intensity = parts[3]

    print(
        f"File: {file.name} | "
        f"Actor: {actor} | "
        f"Emotion: {emotion} | "
        f"Intensity: {intensity}"
    )
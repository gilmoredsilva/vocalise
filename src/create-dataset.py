from pathlib import Path
import pandas as pd

dataset_path = Path("dataset/archive/AudioWAV")

emotion_map = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}

records = []

for file in dataset_path.rglob("*.wav"):
    parts = file.stem.split("_")

    actor = parts[0]
    sentence = parts[1]
    emotion_code = parts[2]
    intensity = parts[3]

    records.append({
        "file": str(file),
        "actor": actor,
        "sentence": sentence,
        "emotion_code": emotion_code,
        "emotion": emotion_map[emotion_code],
        "intensity": intensity
    })

df = pd.DataFrame(records)

df.to_csv("dataset/dataset.csv", index=False)

print("Dataset created successfully!")
print("Total files:", len(df))
print("\nEmotion distribution:")
print(df["emotion"].value_counts())
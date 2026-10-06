import sys
from pathlib import Path

import torch
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "distilhubert")
SAMPLE_RATE = 16000

LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")

# Load feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)

# Load trained model
model = AutoModelForAudioClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()


def predict(audio_path):
    # Load audio
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    # Limit to 8 seconds, same as training
    max_samples = SAMPLE_RATE * 8
    audio = audio[:max_samples]

    # Extract model input
    inputs = feature_extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    predicted_index = torch.argmax(probabilities).item()
    predicted_emotion = LABELS[predicted_index]
    confidence = probabilities[predicted_index].item()

    result = {
        "emotion": predicted_emotion,
        "confidence": float(confidence),
        "probabilities": {
            label: float(probability.item())
            for label, probability in zip(LABELS, probabilities)
        },
    }

    print("\n==============================")
    print(f"Emotion: {predicted_emotion}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("==============================\n")

    print("Probabilities:")

    for label, probability in zip(LABELS, probabilities):
        print(f"{label:<10}: {probability.item() * 100:.2f}%")

    return result


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage:")
        print('python src/predict_distilhubert.py "path/to/audio.wav"')
        sys.exit(1)

    predict(sys.argv[1])
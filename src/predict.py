from pathlib import Path

import joblib
import librosa
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

FRAME_LENGTH_MS = 25
HOP_LENGTH_MS = 15
N_MFCC = 13
PRE_EMPHASIS = 0.97

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression_balanced.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "logistic_regression_balanced_scaler.pkl"


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(audio_file):
    """
    Extract the exact same 54 features used during training.
    """

    signal, sample_rate = librosa.load(
        audio_file,
        sr=None
    )

    # --------------------------------------------------------
    # Pre-emphasis
    # --------------------------------------------------------

    signal = np.append(
        signal[0],
        signal[1:] - PRE_EMPHASIS * signal[:-1]
    )

    # --------------------------------------------------------
    # Frame parameters
    # --------------------------------------------------------

    frame_length = int(
        FRAME_LENGTH_MS / 1000 * sample_rate
    )

    hop_length = int(
        HOP_LENGTH_MS / 1000 * sample_rate
    )

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=N_MFCC,
        n_fft=frame_length,
        hop_length=hop_length
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # --------------------------------------------------------
    # Zero Crossing Rate
    # --------------------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(
        signal,
        frame_length=frame_length,
        hop_length=hop_length
    )

    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    # --------------------------------------------------------
    # RMS Energy
    # --------------------------------------------------------

    energy = librosa.feature.rms(
        y=signal,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    energy_mean = np.mean(energy)
    energy_std = np.std(energy)

    # --------------------------------------------------------
    # Spectral Centroid
    # --------------------------------------------------------

    centroid = librosa.feature.spectral_centroid(
        y=signal,
        sr=sample_rate,
        n_fft=frame_length,
        hop_length=hop_length
    )[0]

    centroid_mean = np.mean(centroid)
    centroid_std = np.std(centroid)

    # --------------------------------------------------------
    # Spectral Bandwidth
    # --------------------------------------------------------

    bandwidth = librosa.feature.spectral_bandwidth(
        y=signal,
        sr=sample_rate,
        n_fft=frame_length,
        hop_length=hop_length
    )[0]

    bandwidth_mean = np.mean(bandwidth)
    bandwidth_std = np.std(bandwidth)

    # --------------------------------------------------------
    # Spectral Rolloff
    # --------------------------------------------------------

    rolloff = librosa.feature.spectral_rolloff(
        y=signal,
        sr=sample_rate,
        n_fft=frame_length,
        hop_length=hop_length
    )[0]

    rolloff_mean = np.mean(rolloff)
    rolloff_std = np.std(rolloff)

    # --------------------------------------------------------
    # Pitch / Fundamental Frequency
    # --------------------------------------------------------

    pitch = librosa.yin(
        signal,
        fmin=65,
        fmax=500,
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length
    )

    pitch = pitch[np.isfinite(pitch)]

    if len(pitch) == 0:
        pitch_mean = 0.0
        pitch_std = 0.0
    else:
        pitch_mean = np.mean(pitch)
        pitch_std = np.std(pitch)

    # --------------------------------------------------------
    # Spectral Contrast
    # --------------------------------------------------------

    contrast = librosa.feature.spectral_contrast(
        y=signal,
        sr=sample_rate,
        n_fft=frame_length,
        hop_length=hop_length
    )

    contrast_mean = np.mean(
        contrast,
        axis=1
    )

    contrast_std = np.std(
        contrast,
        axis=1
    )

    # --------------------------------------------------------
    # Spectral Flux
    # --------------------------------------------------------

    stft = np.abs(
        librosa.stft(
            signal,
            n_fft=frame_length,
            hop_length=hop_length
        )
    )

    stft_norm = stft / (
        np.sum(
            stft,
            axis=0,
            keepdims=True
        ) + 1e-10
    )

    spectral_flux = np.sqrt(
        np.sum(
            np.diff(
                stft_norm,
                axis=1
            ) ** 2,
            axis=0
        )
    )

    spectral_flux_mean = np.mean(
        spectral_flux
    )

    spectral_flux_std = np.std(
        spectral_flux
    )

    # --------------------------------------------------------
    # Combine 54 features
    # --------------------------------------------------------

    features = np.concatenate([
        # 13 MFCC means
        mfcc_mean,

        # 13 MFCC standard deviations
        mfcc_std,

        # ZCR
        [zcr_mean, zcr_std],

        # Energy
        [energy_mean, energy_std],

        # Centroid
        [centroid_mean, centroid_std],

        # Bandwidth
        [bandwidth_mean, bandwidth_std],

        # Rolloff
        [rolloff_mean, rolloff_std],

        # Pitch
        [pitch_mean, pitch_std],

        # 7 spectral contrast means
        contrast_mean,

        # 7 spectral contrast standard deviations
        contrast_std,

        # Spectral flux
        [spectral_flux_mean, spectral_flux_std]
    ])

    return features


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("Model loaded successfully.")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_emotion(audio_file):
    """
    Predict emotion from an audio file.

    Returns:
        emotion
        confidence
        probabilities
    """

    # Extract 54 features
    features = extract_features(audio_file)

    # Check feature count
    if len(features) != 54:
        raise ValueError(
            f"Expected 54 features, got {len(features)}"
        )

    # Convert to 2D array
    features = features.reshape(1, -1)

    # Apply training scaler
    features_scaled = scaler.transform(features)

    # Predict emotion
    prediction = model.predict(
        features_scaled
    )[0]

    # Probability distribution
    probabilities = model.predict_proba(
        features_scaled
    )[0]

    # Get confidence of predicted class
    predicted_index = list(
        model.classes_
    ).index(prediction)

    confidence = probabilities[
        predicted_index
    ]

    # Create probability dictionary
    probability_dict = {
        emotion: float(probability)
        for emotion, probability
        in zip(
            model.classes_,
            probabilities
        )
    }

    return {
        "emotion": prediction,
        "confidence": float(confidence),
        "probabilities": probability_dict
    }


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:
        print(
            "\nUsage:"
        )
        print(
            "python src/predict.py <audio_file>"
        )
        print(
            "\nExample:"
        )
        print(
            "python src/predict.py test.wav"
        )
        sys.exit(1)

    audio_file = sys.argv[1]

    print("\nProcessing:", audio_file)

    result = predict_emotion(audio_file)

    print("\nPrediction")
    print("----------")
    print(
        "Emotion:",
        result["emotion"]
    )

    print(
        "Confidence:",
        f"{result['confidence'] * 100:.2f}%"
    )

    print("\nProbabilities:")

    for emotion, probability in result[
        "probabilities"
    ].items():

        print(
            f"{emotion:10s}: "
            f"{probability * 100:.2f}%"
        )
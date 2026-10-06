from flask import Flask, request, jsonify
from flask_cors import CORS

import sys
import os
import tempfile
import importlib.util
import time

import numpy as np
import librosa


# ============================================================
# Allow Python to find files inside src/
# ============================================================

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src")
    )
)

from predict import predict_emotion


DISTILHUBERT_MODULE = None


def load_distilhubert_predictor():
    """Load the existing predictor once, only when a request needs it."""

    global DISTILHUBERT_MODULE

    if DISTILHUBERT_MODULE is not None:
        return DISTILHUBERT_MODULE

    predictor_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "predict-distilhubert.py"
    )

    spec = importlib.util.spec_from_file_location(
        "vocalise_distilhubert_predictor",
        predictor_path
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the DistilHuBERT predictor.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    DISTILHUBERT_MODULE = module

    return module


def predict_distilhubert(audio_path):
    predictor = load_distilhubert_predictor()
    return predictor.predict(audio_path)


# ============================================================
# Flask setup
# ============================================================

app = Flask(__name__)
CORS(app, origins=os.getenv("CORS_ORIGINS", "*").split(","))


# ============================================================
# DSP configuration
#
# These match the existing Vocalise prediction pipeline.
# DO NOT change these values.
# ============================================================

FRAME_LENGTH_MS = 25
HOP_LENGTH_MS = 15
N_MFCC = 13
PRE_EMPHASIS = 0.97


# ============================================================
# Utility: convert numpy objects to JSON-safe Python objects
# ============================================================

def clean_array(array):
    array = np.asarray(array).astype(float)

    array = np.nan_to_num(
        array,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    return array.tolist()


# ============================================================
# DSP VISUALIZATION ANALYSIS
#
# This does NOT replace predict.py.
#
# It creates visual representations of the SAME processing
# stages so the frontend can show what is happening.
# ============================================================

def analyze_signal(audio_path):

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    y, sr = librosa.load(
        audio_path,
        sr=None,
        mono=True
    )

    if len(y) == 0:
        raise ValueError("Audio file contains no samples.")

    # --------------------------------------------------------
    # Normalize only for visualization
    #
    # The prediction pipeline itself is NOT changed.
    # --------------------------------------------------------

    max_abs = np.max(np.abs(y))

    if max_abs > 0:
        y_visual = y / max_abs
    else:
        y_visual = y

    # --------------------------------------------------------
    # Pre-emphasis
    #
    # Same coefficient used by predict.py
    # y[n] = x[n] - 0.97*x[n-1]
    # --------------------------------------------------------

    emphasized = np.append(
        y[0],
        y[1:] - PRE_EMPHASIS * y[:-1]
    )

    # --------------------------------------------------------
    # Frame parameters
    # --------------------------------------------------------

    frame_length = int(
        sr * FRAME_LENGTH_MS / 1000
    )

    hop_length = int(
        sr * HOP_LENGTH_MS / 1000
    )

    # --------------------------------------------------------
    # STFT
    # --------------------------------------------------------

    stft = librosa.stft(
        emphasized,
        n_fft=2048,
        hop_length=hop_length
    )

    magnitude = np.abs(stft)

    # --------------------------------------------------------
    # Spectrogram
    # --------------------------------------------------------

    power = magnitude ** 2

    spectrogram_db = librosa.power_to_db(
        power,
        ref=np.max
    )

    # Resize/downsample for browser transfer.
    # This is ONLY for visualization.
    target_height = 80
    target_width = 140

    # Time averaging
    if spectrogram_db.shape[1] > target_width:

        indices = np.linspace(
            0,
            spectrogram_db.shape[1],
            target_width + 1
        ).astype(int)

        compressed_time = []

        for i in range(target_width):

            start = indices[i]
            end = indices[i + 1]

            if end <= start:
                end = start + 1

            compressed_time.append(
                np.mean(
                    spectrogram_db[:, start:end],
                    axis=1
                )
            )

        spectrogram_db = np.stack(
            compressed_time,
            axis=1
        )

    # Frequency averaging
    if spectrogram_db.shape[0] > target_height:

        indices = np.linspace(
            0,
            spectrogram_db.shape[0],
            target_height + 1
        ).astype(int)

        compressed_frequency = []

        for i in range(target_height):

            start = indices[i]
            end = indices[i + 1]

            if end <= start:
                end = start + 1

            compressed_frequency.append(
                np.mean(
                    spectrogram_db[start:end, :],
                    axis=0
                )
            )

        spectrogram_db = np.stack(
            compressed_frequency,
            axis=0
        )

    # Normalize spectrogram to 0–1 for canvas rendering

    spec_min = np.min(spectrogram_db)
    spec_max = np.max(spectrogram_db)

    if spec_max > spec_min:

        spectrogram_normalized = (
            (spectrogram_db - spec_min)
            / (spec_max - spec_min)
        )

    else:

        spectrogram_normalized = np.zeros_like(
            spectrogram_db
        )

    # --------------------------------------------------------
    # MFCC
    #
    # Same 13 MFCC configuration used by Vocalise.
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=emphasized,
        sr=sr,
        n_mfcc=N_MFCC,
        hop_length=hop_length
    )

    # Resize MFCC only for visualization.
    # 13 rows remain untouched.

    if mfcc.shape[1] > target_width:

        indices = np.linspace(
            0,
            mfcc.shape[1],
            target_width + 1
        ).astype(int)

        compressed_mfcc = []

        for i in range(target_width):

            start = indices[i]
            end = indices[i + 1]

            if end <= start:
                end = start + 1

            compressed_mfcc.append(
                np.mean(
                    mfcc[:, start:end],
                    axis=1
                )
            )

        mfcc = np.stack(
            compressed_mfcc,
            axis=1
        )

    # Normalize MFCC values for display

    mfcc_min = np.min(mfcc)
    mfcc_max = np.max(mfcc)

    if mfcc_max > mfcc_min:

        mfcc_normalized = (
            (mfcc - mfcc_min)
            / (mfcc_max - mfcc_min)
        )

    else:

        mfcc_normalized = np.zeros_like(mfcc)

    # --------------------------------------------------------
    # Waveform downsampling
    # --------------------------------------------------------

    waveform_points = 1600

    if len(y_visual) > waveform_points:

        indices = np.linspace(
            0,
            len(y_visual),
            waveform_points
        ).astype(int)

        waveform = []

        for i in range(waveform_points):

            start = indices[i]
            end = (
                indices[i + 1]
                if i + 1 < len(indices)
                else len(y_visual)
            )

            if end <= start:
                end = start + 1

            waveform.append(
                np.mean(y_visual[start:end])
            )

        waveform = np.array(waveform)

    else:

        waveform = y_visual

    # --------------------------------------------------------
    # Pre-emphasized waveform downsampling
    # --------------------------------------------------------

    emphasized_visual = emphasized.copy()

    max_emphasis = np.max(
        np.abs(emphasized_visual)
    )

    if max_emphasis > 0:

        emphasized_visual = (
            emphasized_visual / max_emphasis
        )

    if len(emphasized_visual) > waveform_points:

        indices = np.linspace(
            0,
            len(emphasized_visual),
            waveform_points
        ).astype(int)

        emphasized_waveform = []

        for i in range(waveform_points):

            start = indices[i]

            end = (
                indices[i + 1]
                if i + 1 < len(indices)
                else len(emphasized_visual)
            )

            if end <= start:
                end = start + 1

            emphasized_waveform.append(
                np.mean(
                    emphasized_visual[start:end]
                )
            )

        emphasized_waveform = np.array(
            emphasized_waveform
        )

    else:

        emphasized_waveform = emphasized_visual

    # --------------------------------------------------------
    # Return visualization data
    # --------------------------------------------------------

    return {

        "sample_rate": int(sr),

        "duration": float(
            len(y) / sr
        ),

        "frame_length_ms": FRAME_LENGTH_MS,

        "hop_length_ms": HOP_LENGTH_MS,

        "frame_length_samples": frame_length,

        "hop_length_samples": hop_length,

        "waveform": clean_array(waveform),

        "pre_emphasized_waveform":
            clean_array(emphasized_waveform),

        "spectrogram":
            clean_array(spectrogram_normalized),

        "mfcc":
            clean_array(mfcc_normalized),

        "mfcc_count": N_MFCC
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    # --------------------------------------------------------
    # Check upload
    # --------------------------------------------------------

    if "audio" not in request.files:

        return jsonify({
            "error": "No audio file provided"
        }), 400

    audio_file = request.files["audio"]

    if audio_file.filename == "":

        return jsonify({
            "error": "No audio file selected"
        }), 400

    # --------------------------------------------------------
    # Save uploaded audio temporarily
    # --------------------------------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as temp_file:

        audio_path = temp_file.name

        audio_file.save(audio_path)

    try:

        response = {
            "traditional": None,
            "distilhubert": None,
            "comparison": {
                "agreement": None,
                "message": "A comparison is available when both models return a prediction."
            },
            "analysis": None,
            "errors": {}
        }

        # Keep the existing DSP feature extraction and classifier call intact.
        traditional_started = time.perf_counter()

        try:
            traditional = predict_emotion(audio_path)
            traditional["inference_time_ms"] = round(
                (time.perf_counter() - traditional_started) * 1000,
                2
            )
            response["traditional"] = traditional

            # Preserve the original top-level prediction fields for existing clients.
            response.update({
                "emotion": traditional["emotion"],
                "confidence": traditional["confidence"],
                "probabilities": traditional["probabilities"]
            })

        except Exception as error:
            response["errors"]["traditional"] = str(error)

        distilhubert_started = time.perf_counter()

        try:
            distilhubert = predict_distilhubert(audio_path)
            distilhubert["inference_time_ms"] = round(
                (time.perf_counter() - distilhubert_started) * 1000,
                2
            )
            response["distilhubert"] = distilhubert

        except Exception as error:
            response["errors"]["distilhubert"] = str(error)

        if response["traditional"] and response["distilhubert"]:
            traditional_emotion = response["traditional"]["emotion"]
            distilhubert_emotion = response["distilhubert"]["emotion"]
            agreement = traditional_emotion == distilhubert_emotion

            response["comparison"] = {
                "agreement": agreement,
                "message": (
                    "Both models identify the same dominant emotion."
                    if agreement
                    else "The models produce different predictions for this recording."
                )
            }

        try:
            response["analysis"] = analyze_signal(audio_path)
        except Exception as error:
            response["errors"]["analysis"] = str(error)

        if not response["traditional"] and not response["distilhubert"]:
            return jsonify({
                "error": "Both emotion models failed.",
                "details": response["errors"]
            }), 500

        return jsonify(response)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

    finally:

        if os.path.exists(audio_path):

            os.remove(audio_path)


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Vocalise backend is running!"
    })


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1"
    )
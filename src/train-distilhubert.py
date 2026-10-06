from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import Dataset

from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "ntu-spml/distilhubert"

TRAIN_CSV = "dataset/train.csv"
TEST_CSV = "dataset/test.csv"

OUTPUT_DIR = "models/distilhubert"

SAMPLE_RATE = 16000

LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
]

label2id = {
    label: i
    for i, label in enumerate(LABELS)
}

id2label = {
    i: label
    for i, label in enumerate(LABELS)
}


# ============================================================
# DEVICE
# ============================================================

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

print("\n========================================")
print("Vocalise — DistilHuBERT")
print("========================================")
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2
        ),
        "GB"
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

print("Training samples:", len(train_df))
print("Test samples:", len(test_df))

print("\nTraining emotion distribution:")
print(train_df["emotion"].value_counts())

print("\nTest emotion distribution:")
print(test_df["emotion"].value_counts())


# ============================================================
# FEATURE EXTRACTOR
# ============================================================

print("\nLoading feature extractor...")

feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_NAME
)


# ============================================================
# DATASET
# ============================================================

class EmotionDataset(Dataset):

    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        audio_path = row["file"]
        emotion = row["emotion"]

        # Load audio
        import librosa

        audio, sr = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        # Convert waveform to model input
        inputs = feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            max_length=SAMPLE_RATE * 8,
            truncation=True,
        )

        item = {
            "input_values": torch.tensor(
                inputs["input_values"][0],
                dtype=torch.float32
            ),
            "labels": label2id[emotion],
        }

        return item


# ============================================================
# CREATE DATASETS
# ============================================================

print("\nCreating datasets...")

train_dataset = EmotionDataset(train_df)
test_dataset = EmotionDataset(test_df)

print("Datasets ready.")


# ============================================================
# MODEL
# ============================================================

print("\nLoading DistilHuBERT...")

model = AutoModelForAudioClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)

model.to(device)

print("Model loaded.")


# ============================================================
# METRICS
# ============================================================

def compute_metrics(pred):

    predictions = np.argmax(
        pred.predictions,
        axis=1
    )

    labels = pred.label_ids

    accuracy = accuracy_score(
        labels,
        predictions
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro"
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


# ============================================================
# TRAINING
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=1e-5,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    gradient_accumulation_steps=2,

    num_train_epochs=10,

    weight_decay=0.01,

    logging_steps=50,

    load_best_model_at_end=True,

    metric_for_best_model="macro_f1",
    greater_is_better=True,

    fp16=torch.cuda.is_available(),

    save_total_limit=2,

    report_to="none",

    dataloader_num_workers=0,

    remove_unused_columns=False,
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,

    processing_class=feature_extractor,

    compute_metrics=compute_metrics,
)


# ============================================================
# TRAIN
# ============================================================

print("\n========================================")
print("STARTING TRAINING")
print("========================================\n")

trainer.train()


# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n========================================")
print("FINAL TEST RESULTS")
print("========================================")

results = trainer.evaluate()

print("\nAccuracy:")
print(f"{results['eval_accuracy'] * 100:.2f}%")

print("\nMacro F1:")
print(f"{results['eval_macro_f1']:.4f}")


# ============================================================
# SAVE
# ============================================================

print("\nSaving best model...")

trainer.save_model(OUTPUT_DIR)
feature_extractor.save_pretrained(OUTPUT_DIR)

print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")
print("Saved to:", OUTPUT_DIR)
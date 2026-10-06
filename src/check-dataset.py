import pandas as pd

df = pd.read_csv("dataset/dataset.csv")

print("Total samples:", len(df))
print("Total actors:", df["actor"].nunique())

print("\nEmotion distribution:")
print(df["emotion"].value_counts())

print("\nIntensity distribution:")
print(df["intensity"].value_counts())

print("\nSamples per actor:")
print(df["actor"].value_counts().sort_index())
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

audio_features_df = pd.read_csv('audio_features.csv')

# Gráfico: Distribución de características acústicas según "accent"
plt.figure(figsize=(16, 8))
for i, feature in enumerate(["rmse", "spectral_centroid", "mfcc_mean"], 1):
    plt.subplot(1, 3, i)
    sns.boxplot(data=audio_features_df, x="accent", y=feature, palette="pastel")
    plt.title(f"{feature} según Acento", fontsize=12)
    plt.xticks(rotation=45)
    plt.xlabel("Acento")
    plt.ylabel(feature)

plt.tight_layout()
plt.show()

# Gráfico: Distribución de características acústicas según "age" (por grupos de edad)
plt.figure(figsize=(14, 6))
audio_features_df['age_group'] = pd.cut(
    audio_features_df['age'], bins=[0, 20, 40, 60, 80], labels=["0-20", "21-40", "41-60", "61-80"]
)

for i, feature in enumerate(["rmse", "spectral_centroid", "mfcc_mean"], 1):
    plt.subplot(1, 3, i)
    sns.boxplot(data=audio_features_df, x="age_group", y=feature, palette="coolwarm")
    plt.title(f"{feature} según Grupo de Edad", fontsize=12)
    plt.xlabel("Grupo de Edad")
    plt.ylabel(feature)

plt.tight_layout()
plt.show()

# Gráfico: Distribución de características acústicas según "native speaker"
plt.figure(figsize=(14, 6))
for i, feature in enumerate(["rmse", "spectral_centroid", "mfcc_mean"], 1):
    plt.subplot(1, 3, i)
    sns.boxplot(data=audio_features_df, x="native speaker", y=feature, palette="Set3")
    plt.title(f"{feature} según Hablante Nativo", fontsize=12)
    plt.xlabel("Hablante Nativo")
    plt.ylabel(feature)

plt.tight_layout()
plt.show()

# Gráfico: Distribución de características acústicas según "origin" (resumido)
top_origins = audio_features_df['origin'].value_counts().index[:5]
filtered_df = audio_features_df[audio_features_df['origin'].isin(top_origins)]

plt.figure(figsize=(16, 8))
for i, feature in enumerate(["rmse", "spectral_centroid", "mfcc_mean"], 1):
    plt.subplot(1, 3, i)
    sns.boxplot(data=filtered_df, x="origin", y=feature, palette="Set1")
    plt.title(f"{feature} según Origen (Top 5)", fontsize=12)
    plt.xticks(rotation=45)
    plt.xlabel("Origen")
    plt.ylabel(feature)

plt.tight_layout()
plt.show()

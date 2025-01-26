import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# Configuración general de estilo para seaborn
sns.set(style="whitegrid", palette="muted")

audio_features_df = pd.read_csv('audio_features.csv')

# Gráfico: Correlación entre características acústicas
plt.figure(figsize=(10, 8))
correlation_matrix = audio_features_df[
    ["chroma_stft", "rmse", "spectral_centroid", "spectral_bandwidth", "rolloff", "zero_crossing_rate", "mfcc_mean"]
].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Matriz de Correlación entre Características Acústicas", fontsize=14)
plt.show()

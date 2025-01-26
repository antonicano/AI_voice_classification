import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

audio_features_df = pd.read_csv('audio_features.csv')

# Gráfico 1: Distribución de género
plt.figure(figsize=(6, 4))
sns.countplot(data=audio_features_df, x="gender", palette="coolwarm")
plt.title("Distribución por Género", fontsize=14)
plt.xlabel("Género")
plt.ylabel("Cantidad")
plt.show()

# Gráfico 2: Distribución de hablantes nativos
plt.figure(figsize=(6, 4))
sns.countplot(data=audio_features_df, x="native speaker", palette="Set3")
plt.title("Distribución de Hablantes Nativos", fontsize=14)
plt.xlabel("Hablante Nativo")
plt.ylabel("Cantidad")
plt.show()

# Gráfico 3: Distribución por grupos de edad
plt.figure(figsize=(6, 4))
age_groups = pd.cut(
    audio_features_df["age"], bins=[0, 20, 40, 60, 80, 100], labels=["0-20", "21-40", "41-60", "61-80", "81+"]
)
sns.countplot(x=age_groups, palette="viridis")
plt.title("Distribución por Grupo de Edad", fontsize=14)
plt.xlabel("Grupo de Edad")
plt.ylabel("Cantidad")
plt.show()

# Gráfico 4: Distribución por acento
plt.figure(figsize=(10, 6))
sns.countplot(data=audio_features_df, x="accent", order=audio_features_df["accent"].value_counts().index, palette="pastel")
plt.title("Distribución por Acento", fontsize=14)
plt.xticks(rotation=45)
plt.xlabel("Acento")
plt.ylabel("Cantidad")
plt.show()

# Gráfico 5: Distribución por origen (Top 10)
plt.figure(figsize=(10, 6))
top_origins = audio_features_df["origin"].value_counts().nlargest(10)
sns.barplot(x=top_origins.index, y=top_origins.values, palette="Set1")
plt.title("Distribución por Origen (Top 10)", fontsize=14)
plt.xticks(rotation=45)
plt.xlabel("Origen")
plt.ylabel("Cantidad")
plt.show()

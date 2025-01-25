import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Cargar el dataset
print("Cargando datos...")
df = pd.read_csv("audio_features.csv")

# Separar características (X) y etiquetas (y)
X = df.drop(columns=["gender", "native speaker"])  # Eliminar columnas no relevantes
y_native = df["native speaker"]

# Dividir datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y_native, test_size=0.2, random_state=42)

# 2. Entrenar el modelo
print("Entrenando modelo para clasificación de natividad...")
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 3. Evaluar el modelo
y_pred = clf.predict(X_test)

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Crear la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualizar la matriz de confusión
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens", xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.xlabel("Predicción")
plt.ylabel("Verdad")
plt.title("Matriz de Confusión - Natividad")
plt.show()

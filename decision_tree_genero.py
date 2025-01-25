import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Cargar el dataset
print("Cargando datos...")
df = pd.read_csv("audio_features.csv")

# Separar características (X) y etiquetas (y)
X = df.drop(columns=["gender", "native speaker"])  # Eliminar columnas objetivo e irrelevantes
y_gender = df["gender"]

# Convertir datos categóricos en numéricos (One-Hot Encoding para X)
X = pd.get_dummies(X, drop_first=True)

# 2. Dividir datos en entrenamiento y prueba inicial (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y_gender, test_size=0.2, random_state=42)

# 3. Configurar el modelo de Decision Tree con hiperparámetros controlados
clf = DecisionTreeClassifier(
    max_depth=10,  # Evitar árboles muy profundos para reducir sobreajuste
    min_samples_split=5,  # Número mínimo de muestras para dividir un nodo
    random_state=42
)

# 4. Validación cruzada (k=5)
print("Realizando validación cruzada...")
cv_results = cross_validate(
    clf,
    X_train,
    y_train,
    cv=5,
    scoring=["accuracy", "f1_weighted"],  # Métricas de evaluación
    return_train_score=True
)

# Promediar los resultados de validación cruzada
print("Resultados de validación cruzada:")
print(f"Accuracy promedio (validación): {np.mean(cv_results['test_accuracy']):.2f}")
print(f"F1-score promedio (validación): {np.mean(cv_results['test_f1_weighted']):.2f}")

# 5. Entrenar el modelo final en el conjunto completo de entrenamiento
print("Entrenando el modelo final...")
clf.fit(X_train, y_train)

# 6. Evaluar en el conjunto de prueba
y_pred = clf.predict(X_test)

# Calcular métricas finales
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy (prueba): {accuracy:.2f}")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Crear la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualizar la matriz de confusión
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.xlabel("Predicción")
plt.ylabel("Verdad")
plt.title("Matriz de Confusión - Género")
plt.show()

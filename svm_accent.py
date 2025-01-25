import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Cargar el dataset
print("Cargando datos...")
df = pd.read_csv("audio_features.csv")

# 2. Preparar las características (X) y etiquetas (y)
X = df.drop(columns=["number", "person_id", "take", "accent", "origin", "gender", "native speaker"])  # Eliminar irrelevantes
y_accent = df["accent"]

# Normalización/Escalado de las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_accent, test_size=0.2, random_state=42)

# 3. Optimización de hiperparámetros usando GridSearchCV
param_grid = {
    "C": [0.1, 1, 10],
    "gamma": ["scale", 0.1, 0.01, 0.001],
    "kernel": ["rbf"]
}
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Mejor modelo encontrado
best_model = grid_search.best_estimator_
print(f"Mejores hiperparámetros: {grid_search.best_params_}")

# 4. Validación cruzada con el mejor modelo
cv_results = cross_validate(
    best_model,
    X_train,
    y_train,
    cv=5,
    scoring=["accuracy", "f1_weighted"],
    return_train_score=True
)
print(f"Accuracy promedio (validación cruzada): {np.mean(cv_results['test_accuracy']):.2f}")
print(f"F1-score promedio (validación cruzada): {np.mean(cv_results['test_f1_weighted']):.2f}")

# 5. Evaluación en conjunto de prueba
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred)

print("\nEvaluación en conjunto de prueba:")
print(f"Accuracy: {accuracy_test:.2f}")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# 6. Visualización de la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.xlabel("Predicción")
plt.ylabel("Verdad")
plt.title("Matriz de Confusión - Clasificación de Accent")
plt.show()

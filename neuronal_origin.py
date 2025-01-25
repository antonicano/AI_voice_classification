import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Cargar el dataset
print("Cargando datos...")
df = pd.read_csv("audio_features.csv")

# 2. Dividir la columna 'origin' en nuevas columnas
df[["continent", "country", "city"]] = df["origin"].str.split(", ", expand=True)

# 3. Preprocesar los datos
X = df.drop(columns=["number", "person_id", "take", "accent", "origin", "gender", "native speaker", "continent", "country", "city"])  # Características
y_origin = df["origin"]

# Convertir etiquetas categóricas en números
label_encoder = LabelEncoder()
y_origin_encoded = label_encoder.fit_transform(y_origin)
y_origin_categorical = to_categorical(y_origin_encoded)  # Codificación One-Hot para Keras

# Normalizar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_origin_categorical, test_size=0.2, random_state=42)

# 4. Crear el modelo de red neuronal
model = Sequential([
    Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
    Dropout(0.3),  # Evitar sobreajuste
    Dense(64, activation='relu'),
    Dropout(0.3),  # Evitar sobreajuste
    Dense(y_origin_categorical.shape[1], activation='softmax')  # Salida con tantas clases como valores únicos de 'origin'
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Entrenar el modelo
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# 6. Evaluación en conjunto de prueba
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nPérdida en prueba: {loss:.4f}")
print(f"Precisión en prueba: {accuracy:.4f}")

# 7. Generar métricas
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

print("\nReporte de Clasificación:")
print(classification_report(y_test_classes, y_pred_classes, target_names=label_encoder.classes_))

# Matriz de confusión
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicción")
plt.ylabel("Verdad")
plt.title("Matriz de Confusión - Clasificación de Origin")
plt.show()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# 1. Cargar los datos desde el CSV
print("Cargando datos...")
df = pd.read_csv("audio_features_modificado.csv")

# 2. Preprocesar los datos
print("Preprocesando datos...")

# Separar características (X) y etiquetas (y)
X = df.drop(columns=["gender", "accent", "origin", "native speaker", "age"])  # Características
y = {
    "gender_output": df["gender"],  # Clasificación por género
    "accent_output": df["accent"],  # Clasificación por acento
    "origin_output": df["origin"],  # Clasificación por origen
    "native_output": df["native speaker"],  # Clasificación por nativo/no nativo
    "age_output": df["age"],  # Clasificación por edad
}

# Normalizar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Codificar las etiquetas (LabelEncoder para convertir texto a números)
le_gender = LabelEncoder()
le_accent = LabelEncoder()
le_origin = LabelEncoder()
le_native = LabelEncoder()
le_age = LabelEncoder()

y["gender_output"] = le_gender.fit_transform(y["gender_output"])
y["accent_output"] = le_accent.fit_transform(y["accent_output"])
y["origin_output"] = le_origin.fit_transform(y["origin_output"])
y["native_output"] = le_native.fit_transform(y["native_output"])
y["age_output"] = le_age.fit_transform(y["age_output"])

# Convertir etiquetas a formato categórico (one-hot encoding)
y["gender_output"] = to_categorical(y["gender_output"])
y["accent_output"] = to_categorical(y["accent_output"])
y["origin_output"] = to_categorical(y["origin_output"])
y["native_output"] = to_categorical(y["native_output"])
y["age_output"] = to_categorical(y["age_output"])

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Crear la red neuronal
print("Creando modelo...")
model = Sequential()

# Capa de entrada
model.add(Dense(128, input_dim=X_scaled.shape[1], activation="relu"))
model.add(Dropout(0.3))  # Evitar overfitting

# Capa oculta
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))

# Salida múltiple para cada categoría
model.add(Dense(y["gender_output"].shape[1], activation="softmax", name="gender_output"))
model.add(Dense(y["accent_output"].shape[1], activation="softmax", name="accent_output"))
model.add(Dense(y["origin_output"].shape[1], activation="softmax", name="origin_output"))
model.add(Dense(y["native_output"].shape[1], activation="softmax", name="native_output"))
model.add(Dense(y["age_output"].shape[1], activation="softmax", name="age_output"))

# Compilar el modelo
print("Compilando modelo...")
model.compile(optimizer=Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# 4. Entrenar el modelo
print("Entrenando modelo...")
history = model.fit(
    X_train,
    {"gender_output": y_train["gender_output"],
     "accent_output": y_train["accent_output"],
     "origin_output": y_train["origin_output"],
     "native_output": y_train["native_output"],
     "age_output": y_train["age_output"]},
    epochs=50,
    batch_size=32,
    validation_data=(X_test, {
        "gender_output": y_test["gender_output"],
        "accent_output": y_test["accent_output"],
        "origin_output": y_test["origin_output"],
        "native_output": y_test["native_output"],
        "age_output": y_test["age_output"]
    })
)

# 5. Evaluar el modelo
print("Evaluando modelo...")
losses = model.evaluate(
    X_test,
    {"gender_output": y_test["gender_output"],
     "accent_output": y_test["accent_output"],
     "origin_output": y_test["origin_output"],
     "native_output": y_test["native_output"],
     "age_output": y_test["age_output"]}
)
print(f"Resultados de evaluación: {losses}")

# 6. Guardar el modelo
print("Guardando modelo...")
model.save("audio_classification_model.h5")
print("Modelo guardado como 'audio_classification_model.h5'")

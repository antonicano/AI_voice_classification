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
y_gender = df["gender"]  # Clasificación por género
y_accent = df["accent"]  # Clasificación por acento
y_origin = df["origin"]  # Clasificación por origen
y_native = df["native speaker"]  # Clasificación por nativo/no nativo
y_age = df["age"]  # Clasificación por edad

# Normalizar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Codificar las etiquetas (LabelEncoder para convertir texto a números)
le_gender = LabelEncoder()
le_accent = LabelEncoder()
le_origin = LabelEncoder()
le_native = LabelEncoder()
le_age = LabelEncoder()

y_gender_enc = le_gender.fit_transform(y_gender)
y_accent_enc = le_accent.fit_transform(y_accent)
y_origin_enc = le_origin.fit_transform(y_origin)
y_native_enc = le_native.fit_transform(y_native)
y_age_enc = le_age.fit_transform(y_age)

# Convertir etiquetas a formato categórico (one-hot encoding)
y_gender_cat = to_categorical(y_gender_enc)
y_accent_cat = to_categorical(y_accent_enc)
y_origin_cat = to_categorical(y_origin_enc)
y_native_cat = to_categorical(y_native_enc)
y_age_cat = to_categorical(y_age_enc)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train_gender, y_test_gender = train_test_split(X_scaled, y_gender_cat, test_size=0.2, random_state=42)
_, _, y_train_accent, y_test_accent = train_test_split(X_scaled, y_accent_cat, test_size=0.2, random_state=42)
_, _, y_train_origin, y_test_origin = train_test_split(X_scaled, y_origin_cat, test_size=0.2, random_state=42)
_, _, y_train_native, y_test_native = train_test_split(X_scaled, y_native_cat, test_size=0.2, random_state=42)
_, _, y_train_age, y_test_age = train_test_split(X_scaled, y_age_cat, test_size=0.2, random_state=42)

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
outputs = []
outputs.append(Dense(y_gender_cat.shape[1], activation="softmax", name="gender_output"))
outputs.append(Dense(y_accent_cat.shape[1], activation="softmax", name="accent_output"))
outputs.append(Dense(y_origin_cat.shape[1], activation="softmax", name="origin_output"))
outputs.append(Dense(y_native_cat.shape[1], activation="softmax", name="native_output"))
outputs.append(Dense(y_age_cat.shape[1], activation="softmax", name="age_output"))

for output in outputs:
    model.add(output)

# Compilar el modelo
print("Compilando modelo...")
model.compile(optimizer=Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# 4. Entrenar el modelo
print("Entrenando modelo...")
history = model.fit(
    X_train,
    {"gender_output": y_train_gender,
     "accent_output": y_train_accent,
     "origin_output": y_train_origin,
     "native_output": y_train_native,
     "age_output": y_train_age},
    epochs=50,
    batch_size=32,
    validation_data=(X_test, {
        "gender_output": y_test_gender,
        "accent_output": y_test_accent,
        "origin_output": y_test_origin,
        "native_output": y_test_native,
        "age_output": y_test_age
    })
)

# 5. Evaluar el modelo
print("Evaluando modelo...")
losses = model.evaluate(
    X_test,
    {"gender_output": y_test_gender,
     "accent_output": y_test_accent,
     "origin_output": y_test_origin,
     "native_output": y_test_native,
     "age_output": y_test_age}
)
print(f"Resultados de evaluación: {losses}")

# 6. Guardar el modelo
print("Guardando modelo...")
model.save("audio_classification_model.h5")
print("Modelo guardado como 'audio_classification_model.h5'")

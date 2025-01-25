import os
import json
import librosa
import pandas as pd

# Ruta principal
DATA_FOLDER = "data"
META_FILE = os.path.join(DATA_FOLDER, "audioMNIST_meta.txt")
LOG_FILE = "log.txt"

# Función para cargar el archivo meta
def load_metadata(meta_file):
    with open(meta_file, "r") as f:
        metadata = json.load(f)
    return metadata

# Función para extraer características de audio usando librosa
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        features = {
            "chroma_stft": librosa.feature.chroma_stft(y=y, sr=sr).mean(),
            "rmse": librosa.feature.rms(y=y).mean(),
            "spectral_centroid": librosa.feature.spectral_centroid(y=y, sr=sr).mean(),
            "spectral_bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=sr).mean(),
            "rolloff": librosa.feature.spectral_rolloff(y=y, sr=sr).mean(),
            "zero_crossing_rate": librosa.feature.zero_crossing_rate(y=y).mean(),
            "mfcc_mean": librosa.feature.mfcc(y=y, sr=sr).mean(),
            "mfcc_var": librosa.feature.mfcc(y=y, sr=sr).var(),
        }
        return features
    except Exception as e:
        log_error(f"Error processing {file_path}: {e}")
        return None

# Función para registrar errores y procesos
def log_error(message):
    with open(LOG_FILE, "a") as log:
        log.write(message + "\n")

# Cargar metadatos
metadata = load_metadata(META_FILE)

# Listado de datos procesados
data = []

# Recorrer las carpetas numeradas (01 a 60)
for folder in sorted(os.listdir(DATA_FOLDER)):
    folder_path = os.path.join(DATA_FOLDER, folder)
    if os.path.isdir(folder_path):
        # Recorrer los archivos dentro de la carpeta
        for file_name in sorted(os.listdir(folder_path)):
            if file_name.endswith(".wav"):
                file_path = os.path.join(folder_path, file_name)

                # Extraer información del nombre del archivo
                parts = file_name.replace(".wav", "").split("_")
                if len(parts) == 3:
                    n = parts[0]  # Número hablado (0-9)
                    pp = parts[1]  # Clasificación de la persona (01-60)
                    t = parts[2]  # Toma (0-49)

                    # Obtener datos de la persona desde los metadatos
                    person_meta = metadata.get(pp, {})

                    # Extraer características de audio
                    features = extract_features(file_path)
                    if features:
                        data.append({
                            "number": n,
                            "person_id": pp,
                            "take": t,
                            **features,
                            **person_meta
                        })
                        log_error(f"Processed: {file_name}")
                    else:
                        log_error(f"Failed to extract features: {file_name}")

# Crear un DataFrame con los datos procesados
df = pd.DataFrame(data)

# Guardar los datos en un archivo CSV
df.to_csv("audio_features.csv", index=False)

print("Exportación completada. Archivo guardado como audio_features.csv")

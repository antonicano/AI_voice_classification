import pandas as pd

# Cargar el CSV en un DataFrame
df = pd.read_csv('audio_features.csv')

# Eliminar las columnas 'recordingdate' y 'recordingroom'
df = df.drop(columns=['recordingdate', 'recordingroom'])

# Guardar el DataFrame modificado en un nuevo archivo CSV
df.to_csv('audio_features_modificado.csv', index=False)
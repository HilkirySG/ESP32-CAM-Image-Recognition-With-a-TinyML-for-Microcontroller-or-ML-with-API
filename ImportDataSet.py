import os
import tensorflow as tf # Ensure tf is imported for tf.keras.utils.get_file

print("Descargando ZIP de datos")
url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
# tf.keras.utils.get_file extracts the content into a temporary directory.
# This function returns the path to that temporary extraction directory.
path_to_zip_extraction_root = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin=url, extract=True)

# Variables para rutas en disco
# The actual dataset content (train/validation folders) is inside 'cats_and_dogs_filtered'
# which itself is inside the 'path_to_zip_extraction_root'.
carpeta_base = os.path.join(path_to_zip_extraction_root, 'cats_and_dogs_filtered')
carpeta_entrenamiento = os.path.join(carpeta_base, 'train')
carpeta_validacion = os.path.join(carpeta_base, 'validation')

carp_entren_gatos = os.path.join(carpeta_entrenamiento, 'cats')  # imagenes de gatos para pruebas
carpeta_entren_perros = os.path.join(carpeta_entrenamiento, 'dogs')  # imagenes de perros para pruebas
carpeta_val_gatos = os.path.join(carpeta_validacion, 'cats')  # imagenes de gatos para validacion
carpeta_val_perros = os.path.join(carpeta_validacion, 'dogs')  # imagenes de perros para validacion

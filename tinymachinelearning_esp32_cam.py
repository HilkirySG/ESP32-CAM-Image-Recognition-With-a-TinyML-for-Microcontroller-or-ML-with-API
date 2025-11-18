
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers, models

# --- 1. Definición de Parámetros ---
IMG_HEIGHT = 64
IMG_WIDTH = 64
IMG_CHANELS = 1 # Monocromo
BATCH_SIZE = 32

# --- 2. Carga de Datos Eficiente (El método recomendado) ---

# Carga los datos pidiendo 64x64 y monocromático directamente
train_ds = tf.keras.utils.image_dataset_from_directory(
    carpeta_entrenamiento,
    labels='inferred',
    label_mode='binary', # 'binary' es para Sigmoid (etiquetas 0 o 1)
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    carpeta_validacion,
    labels='inferred',
    label_mode='binary',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Optimiza la carga
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


# --- 3. Definición del Modelo ---

data_argumentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_argumentation")

model = models.Sequential([
    # Capa de entrada. Especifica la forma Y normaliza los datos.
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANELS)),
    layers.Rescaling(1./255), # Capa de normalización

    # Capa Convolucional pequeña
    layers.Conv2D(8, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    # Segunda capa opcional (ayuda a la precisión sin añadir mucho tamaño)
    layers.Conv2D(16, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),

    layers.Dropout(0.5),
    # Capa Densa oculta pequeña
    layers.Dense(16, activation='relu'),

    # Capa de Salida (1 neurona, Sigmoid para binario)
    layers.Dense(1, activation='sigmoid') # <-- ¡TU PETICIÓN ORIGINAL!
])

model.summary()

# --- 4. Compilación del Modelo ---
model.compile(optimizer='adam',
              loss='binary_crossentropy', # <-- ¡La pérdida correcta para Sigmoid!
              metrics=['accuracy'])

#Define el "calback" de parada temprana
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', #Monitorea la pérdida de validación
    patience=5, #Espera 5 épocas antes de parar
    restore_best_weights=True #Guarda el mejor modelo, no el último
)

# --- 5. Entrenamiento ---

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Entrena el modelo
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[early_stopping]
)


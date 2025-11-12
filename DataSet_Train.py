
import tensorflow_datasets as tfds
#Carga de dataset 'cats_vs_dogs
# 'Split' divide los datos: 80% para entrenamiento , 10% para validación,
#10% de prueba

(ds_train, ds_val, ds_test), ds_info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info= True, #Para obtener información del dataset
    as_supervised=True, #Devuelve (imagen, etiqueta) en tuplas
)

print(f"Total de imagenes de entrenamiento: {tfds.experimental.cardinality(ds_train)}")

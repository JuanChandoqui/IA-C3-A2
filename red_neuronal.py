from cProfile import label
import zipfile
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import numpy as np

local_zip = './Zips/rps-training.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('./tmp')
zip_ref.close()

local_zip = './Zips/rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('./tmp')
zip_ref.close()

local_zip = './Zips/rps-validation.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('./tmp/rps-validation')
zip_ref.close()

#DIMENSIONES DE LAS IMAGENES
width = 150
height = 150

width_2 = 28
heigth_2 = 28


#VALORES DE ENTRADAS
directorio_entrenamiento = "./tmp/rps/"
generador_de_imagenes = ImageDataGenerator(rescale = 1./255)
generador_entrenamiento = generador_de_imagenes.flow_from_directory(
    directorio_entrenamiento,
    target_size= (width,height),
    class_mode = 'categorical',
    batch_size=126
)

#VALORES DE SALIDA
directorio_entrenamiento = "./tmp/rps-test-set/"
generador_de_imagenes = ImageDataGenerator(rescale =1./255)
generador_validaciones = generador_de_imagenes.flow_from_directory(
    directorio_entrenamiento,
    target_size= (width,height),
    class_mode = 'categorical',
    batch_size=126
)


#VALORES DE ENTRADAS
directorio_entrenamiento = "./tmp/rps/"
generador_entrenamiento_2 = tf.keras.preprocessing.image_dataset_from_directory(
    directorio_entrenamiento, 
    color_mode="grayscale",  
    image_size=(28, 28),
    label_mode='categorical',
    batch_size=8
)

#VALORES DE SALIDA
directorio_entrenamiento = "./tmp/rps-test-set/"
generador_validaciones_2 = tf.keras.preprocessing.image_dataset_from_directory(
    directorio_entrenamiento, 
    color_mode="grayscale", 
    image_size=(28, 28),
    label_mode='categorical',
    batch_size=8
)

image_evaluate = "./tmp/rps-validation/rock1.png"

print(f'START MODEL 1 IN {datetime.datetime.now()}')
#MODELO CON COLORES (RGB)
model_1 = tf.keras.models.Sequential([
    #INPUT_SHAPE SE DEFINE LAS DIMENSIONES DE LAS IMAGENES, RESPETANDO EL COLOR EN RGB (3 BYTES DE COLORES)
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(width,height, 3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    # #mejorar la eficiencia de la red neuronal, eliminando información basura
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),

    #aplicación de la capa DENSE
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model_1.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
history_1 = model_1.fit_generator(generador_entrenamiento, steps_per_epoch=20, epochs=20, validation_data=generador_validaciones, verbose=True, validation_steps=3)

print(f'FINISH MODEL 1 IN {datetime.datetime.now()}')

img = tf.keras.preprocessing.image.load_img(image_evaluate, target_size=(width,height))
x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
result_1 = model_1.predict(images, batch_size=10)
print(result_1)

print(f'START MODEL 2 IN {datetime.datetime.now()}')
# MODELO A ESCALA GRIS
model_2 = tf.keras.models.Sequential([
    #INPUT_SHAPE SE DEFINE LAS DIMENSIONES DE LAS IMAGENES, RESPETANDO EL COLOR EN BLANCO Y NEGRO (1)
    # tf.keras.layers.Conv2D(32, (3,3), input_shape=(28,28, 1),  activation='relu'),
    # tf.keras.layers.MaxPooling2D(2,2),

    # #mejorar la eficiencia de la red neuronal, eliminando información basura
    tf.keras.layers.Flatten(input_shape=(28,28, 1)),
    # tf.keras.layers.Dropout(0.5),

    #aplicación de la capa DENSE
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model_2.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
history_2 = model_2.fit_generator(generador_entrenamiento_2.repeat(), steps_per_epoch=20, epochs=100, validation_data=generador_validaciones_2.repeat(), verbose=True, validation_steps=3)
print(f'FINISH MODEL 2 IN {datetime.datetime.now()}')

img = tf.keras.preprocessing.image.load_img(
    image_evaluate,
    color_mode='grayscale',
    target_size=(28, 28)
)
x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
result_2 = model_2.predict(images, batch_size=10)

print(f'RESULTADO 2 : {result_2}')
loss_1 = history_1.history['loss']
loss_2 = history_2.history['loss']
val_loss = history_1.history['val_loss']
acc = history_1.history['accuracy']
val_acc = history_1.history['val_accuracy']

epochs = range(len(acc))

plt.plot(loss_1, label="LOSS RGB")
plt.plot(loss_2, label="LOSS GRAY")
plt.legend()
plt.xlabel("iteraciones")
plt.ylabel("errores")
plt.show()


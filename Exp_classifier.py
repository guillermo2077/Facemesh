import pandas as pd
import numpy as np
import os
import seaborn as sns

import matplotlib.pyplot as plt
import random

from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from keras.callbacks import ReduceLROnPlateau
from keras_preprocessing.image import ImageDataGenerator
from keras import models


# from keras.optimizers import


def prepare_data(data_array):
    """ Prepare data for modeling
        input: data frame with labels und pixel data
        output: image and label array """

    image_array = np.zeros(shape=(len(data_array), 48, 48))
    image_label = np.array(list(map(int, data_array['emotion'])))

    for i, row in enumerate(data_array.index):
        image = np.fromstring(data_array.loc[row, ' pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))
        image_array[i] = image

    return image_array, image_label


def sample_plot(x, y=None):
    # x, y are numpy arrays
    n = 20
    samples = random.sample(range(x.shape[0]), n)

    fig, axs = plt.subplots(2, 10, figsize=(25, 5), sharex=True, sharey=True)
    ax = axs.ravel()
    for i in range(n):
        ax[i].imshow(x[samples[i], :, :], cmap=plt.get_cmap('gray'))
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        if y is not None:
            ax[i].set_title(emotions[y[samples[i]]])

    plt.show()


# lectura del csv
data = pd.read_csv('assets//icml_face_data.csv')
# print(data.info())

emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# datos imbalanceados, como se muestra en el plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
sns.countplot(data=data[data[' Usage'] == 'Training'], x='emotion', ax=ax1).set_title('Training')
ax1.set_xticklabels(emotions.values())
sns.countplot(data=data[data[' Usage'] == 'PublicTest'], x='emotion', ax=ax2).set_title('Testing')
ax2.set_xticklabels(emotions.values())
sns.countplot(data=data[data[' Usage'] == 'PrivateTest'], x='emotion', ax=ax3).set_title('Validation')
ax3.set_xticklabels(emotions.values())

# plt.show()
# plt.close('all')

# preparamos arrays a partir de el dataset, gracias a la funcion prepare data el nuevo array no tiene la columna usage
# image_array shape = (entradas, 48, 48)
# image_label shape = (entradas,)
train_image_array, train_image_label = prepare_data(data[data[' Usage'] == 'Training'])
val_image_array, val_image_label = prepare_data(data[data[' Usage'] == 'PrivateTest'])
test_image_array, test_image_label = prepare_data(data[data[' Usage'] == 'PublicTest'])

print(train_image_array.shape)
# print(train_image_label.shape)

# ahora adaptamos las labels para la entrada a la red neuronal -> las convertimos en matrices binarias de dummies
train_labels_nn = np.eye(7)[train_image_label]
test_labels_nn = np.eye(7)[test_image_label]
val_labels_nn = np.eye(7)[val_image_label]

print(train_labels_nn.shape)

sample_plot(train_image_array, train_image_label)

# especificamos shape, equivalente a la anterior, preparamos el valor de grayscale para la entrada a la red neuronal,
# el resultado seran matrices con valores de grayscale entre 0 y 1 (importante para la red) con la forma
# (entradas, 48, 48, 1)
train_images = train_image_array.reshape((train_image_array.shape[0], 48, 48, 1))
train_images = train_images.astype('float32') / 255
val_images = val_image_array.reshape((val_image_array.shape[0], 48, 48, 1))
val_images = val_images.astype('float32') / 255
test_images = test_image_array.reshape((test_image_array.shape[0], 48, 48, 1))
test_images = test_images.astype('float32') / 255

# print(train_images.shape)
# print(train_image_label[:10])
# sample_plot(train_images, train_image_label)

shape_imp = train_images.shape[1:]
# print(shape_to_input)

################
# RED NEURONAL #
################

model = models.Sequential()
# modelo secuencial
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPool2D((2, 2)))
# capa convolucion 2D
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D((2, 2)))
# Capa convolucion 2D
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(7, activation='softmax'))
# Capa de finalizacion 7 nodos de salida 

model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

################
# RED NEURONAL #
################

epochs = 27
batch_size = 32

# generamos imagenes a partir de las que tenemos rotadolas o haciendo zoom, etc

model.summary()

#History = model.fit(train_images, train_labels_nn,
                    #epochs=epochs, validation_data=(test_images, test_labels_nn),
                    #batch_size=batch_size, verbose=1)

###################
# EXPORTAR MODELO #
###################

# Guardar el Modelo
#model.save('assets//exp_classifier.h5')

# Recrea exactamente el mismo modelo solo desde el archivo
# new_model = keras.models.load_model('assets//exp_classifier.h5')
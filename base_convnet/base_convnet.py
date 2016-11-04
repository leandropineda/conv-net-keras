'''Trains a simple convnet on the IAM Handwriting Database dataset.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

batch_size = 32
nb_classes = 91
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# dataset path
# It should contain one subdirectory per class, and 
# the subdirectories should contain PNG or JPG images.
trn_path = "../dataset/data_dest_dir/train/chars/"
val_path = "../dataset/data_dest_dir/validation/chars/"
tst_path = "../dataset/data_dest_dir/test/chars/"

# En este generador metemos el aumentado.
# TODO: Ojo que el generador de Train solo tiene que tener aumentado (Hay que hacer un generator para train, y un generator pelado para valid/test)
img_generator = ImageDataGenerator(
                    rescale=1./255,
                    samplewise_center=True,
                    samplewise_std_normalization=True
                    )

# Viendo que hay dos formas que nadie se pone de acuerdo, tomo este orden que pusiste acá: validación para elegir hiperparámetros, test para el test final.

trn_dataset = img_generator.flow_from_directory(
    trn_path,
    target_size=(img_rows,img_cols),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size
)
val_dataset = img_generator.flow_from_directory(
    val_path,
    taget_size=(img_rows,img_cols),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size
)
tst_dataset = img_generator.flow_from_directory(
    tst_path,
    target_size=(img_rows,img_cols),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size
)

trainSamples = trn_dataset.N
validationSamples = val_dataset.N
testSamples = tst_dataset.N

input_shape = trn_dataset.image_shape

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta', # alternativa: rmsprop?
              metrics=['accuracy'])

history = model.fit_generator(trn_dataset, 
                                nb_epoch=nb_epoch,
                                samples_per_epoch=trainSamples,
                                validation_data=val_dataset,
                                nb_val_samples=validationSamples,
                                verbose=1) # nunca me termino de convencer esta implemntación...Entiendo que si no vamos a hacer algo como early stop, no necesitamos pasar validación acá. .
trainAcc = history.history['acc'][-1]


# Aca hay que elegir los mejores hiperparametros, reentrenar con train+validación (hay que hacer otro generador...no se si hace falta dado que son muchos datos), y despues se evalua en test

scoreT = model.evaluate_generator(tst_dataset, testSamples, verbose=0)
print('Test score:', scoreT[0])
print('Test accuracy:', scoreT[1])

print(history.history)

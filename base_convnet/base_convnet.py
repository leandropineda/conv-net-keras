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
nb_classes = 10
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
                    rescale=1./255
                    )

# Viendo que hay dos formas que nadie se pone de acuerdo, tomo este orden que pusiste acá: validación para elegir hiperparámetros, test para el test final.

trn_dataset = img_generator.flow_from_directory(
    trn_path,
    batch_size=batch_size
)
val_dataset = img_generator.flow_from_directory(
    val_path,
    batch_size=batch_size
)
tst_dataset = img_generator.flow_from_directory(
    tst_path,
    batch_size=batch_size
)

trainSamples=trn_dataset.N
validationSamples=val_dataset.N
testSamples=tst_dataset.N


# TODO: hay que resolver el tema de las clases. Creo que por defecto estaría en clases categórcas.


# Desde aca es mnist: the data, shuffled and split between train and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data() # TODO: esto es del mnist no? Yo digo que hagamos todo usando generators que despues vamos a neceistar... 

# if K.image_dim_ordering() == 'th':
#     X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#     X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
#     X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)

# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255
# print('X_train shape:', X_train.shape)
# print(X_train.shape[0], 'train samples')
# print(X_test.shape[0], 'test samples')

# # convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)

# Hasta aca parece que es todo mnist

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
model.add(Dense(91))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta', # alternativa: rmsprop?
              metrics=['accuracy'])

history = model.fit_generator(trn_dataset, 
                                nb_epoch=nb_epoch,
                                samples_per_epoch=trainSamples,
                                verbose=1) # nunca me termino de convencer esta implemntación...Entiendo que si no vamos a hacer algo como early stop, no necesitamos pasar validación acá. .
trainAcc=history.history['acc'][-1]
                    

# Error de validación: para optimizar hiperparámetros
scoreV = model.evaluate(val_dataset,validationSamples, verbose=0)
print('Valid score:', scoreV[0])
print('Valid accuracy:', scoreV[1])


# Aca hay que elegir los mejores hiperparametros, reentrenar con train+validación (hay que hacer otro generador...no se si hace falta dado que son muchos datos), y despues se evalua en test

scoreV = model.evaluate_generator(tst_dataset,testSamples, verbose=0)
print('Test score:', scoreT[0])
print('Test accuracy:', scoreT[1])

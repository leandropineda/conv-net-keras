'''Trains a simple convnet on the IAM Handwriting Database dataset.
'''

from __future__ import print_function
import numpy as np
#np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import json
import pprint

batch_size = 32
nb_classes = 91
nb_epoch = 15

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

####################################################################################################################
# Dataset path
####################################################################################################################
# It should contain one subdirectory per class, and 
# the subdirectories should contain PNG or JPG images.
####################################################################################################################
trn_path = "../dataset/data_dest_dir/train/chars/"
val_path = "../dataset/data_dest_dir/validation/chars/"
tst_path = "../dataset/data_dest_dir/test/chars/"

# En este generador metemos el aumentado.
# TODO: Ojo que el generador de Train solo tiene que tener aumentado (Hay que hacer un generator para train, y un generator pelado para valid/test)
img_generator = ImageDataGenerator(
                    #rescale=1./255,
                    #samplewise_center=True,
                    #samplewise_std_normalization=True
                    )

# Viendo que hay dos formas que nadie se pone de acuerdo, tomo este orden que pusiste acá: validación para elegir hiperparámetros, test para el test final.

trn_dataset = img_generator.flow_from_directory(
    trn_path,
    target_size=(img_rows, img_cols),
    #color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size
)
val_dataset = img_generator.flow_from_directory(
    val_path,
    target_size=(img_rows, img_cols),
    #color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size
)
tst_dataset = img_generator.flow_from_directory(
    tst_path,
    target_size=(img_rows, img_cols),
    #color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size
)

trainSamples = trn_dataset.N
validationSamples = val_dataset.N
testSamples = tst_dataset.N

input_shape = trn_dataset.image_shape
print("Image shape: " + str(input_shape))

####################################################################################################################
# Build the model
####################################################################################################################
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
              optimizer='adadelta',  # alternativa: rmsprop?
              metrics=['accuracy'])

model.summary()

####################################################################################################################
# Save the model to a json file
####################################################################################################################
text_file = open('model.json', 'w')
model_json = str(model.to_json())
parsed_model_json = json.loads(model_json)
text_file.write(json.dumps(parsed_model_json,  # JSON pretty print
                           sort_keys=True,
                           indent=4))
text_file.close()

####################################################################################################################
# Start training
####################################################################################################################
history = model.fit_generator(trn_dataset,
                                nb_epoch=nb_epoch,
                                samples_per_epoch=trainSamples,
                                validation_data=val_dataset,
                                nb_val_samples=validationSamples)

# Aca hay que elegir los mejores hiperparametros, reentrenar con train+validación (hay que hacer otro generador...no se si hace falta dado que son muchos datos), y despues se evalua en test

scoreT = model.evaluate_generator(tst_dataset, testSamples)
print("Test score: " + str(scoreT[0]) + "\nTest accuracy: " + str(scoreT[1]))

####################################################################################################################
# Save some results to a text file
####################################################################################################################
text_file = open('results.txt', 'w')
text_file.write('Test score: ' + str(scoreT[0]))
text_file.write('\nTest accuracy: ' + str(scoreT[1]))
text_file.write('\n\nNumber of parameters: ' + str(model.count_params()))
text_file.write('\n\nImage shape: ' + str(input_shape))
text_file.write('\n\n')
pp = pprint.PrettyPrinter(indent=4)  # dictionary pretty print
text_file.write(pp.pformat(history.history))
text_file.close()

####################################################################################################################
# Plot some results and save to file
####################################################################################################################
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig("results_acc.png")
# summarize history for loss
plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig("results_loss.png")

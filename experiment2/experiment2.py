'''Trains a simple convnet on the IAM Handwriting Database dataset.
'''

from __future__ import print_function
import numpy as np
#np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, SpatialDropout2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import matplotlib.pyplot as plt
import json
import pprint
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):


    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = 50.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="red" if cm[i, j] > thresh and i != j else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

batch_size = 32
nb_classes = 91
nb_epoch = 40

# input image dimensions
img_rows, img_cols = 32, 32
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
trn_path = "../dataset/dataset_preproc/train/chars/"
val_path = "../dataset/dataset_preproc/validation/chars/"
tst_path = "../dataset/dataset_preproc/test/chars/"


# En este generador metemos el aumentado.
# TODO: Ojo que el generador de Train solo tiene que tener aumentado (Hay que hacer un generator para train, y un generator pelado para valid/test)
img_generator = ImageDataGenerator(
                    rescale=1./255
                    )

# Viendo que hay dos formas que nadie se pone de acuerdo, tomo este orden que pusiste acá: validación para elegir hiperparámetros, test para el test final.

trn_dataset = img_generator.flow_from_directory(
    trn_path,
    target_size=(img_rows, img_cols),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size
    #save_to_dir='/home/lpineda/img_test/'  # OJO: escribe muchas imagenes a disco
)
val_dataset = img_generator.flow_from_directory(
    val_path,
    target_size=(img_rows, img_cols),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size
)
tst_dataset = img_generator.flow_from_directory(
    tst_path,
    target_size=(img_rows, img_cols),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size
)

trn_samples = int(trn_dataset.N)
val_samples = int(val_dataset.N)
tst_samples = int(tst_dataset.N)

input_shape = (32, 32, 1)
print("Image shape: " + str(input_shape))

####################################################################################################################
# Build the model
####################################################################################################################
model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        border_mode='same',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(BatchNormalization()) # delete this

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='same'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=pool_size))

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='same'))
model.add(Activation('relu'))

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='same'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=pool_size))

model.add(Convolution2D(nb_filters*2, kernel_size[0], kernel_size[1], border_mode='same'))
model.add(Activation('relu'))

model.add(Convolution2D(nb_filters*2, kernel_size[0], kernel_size[1], border_mode='same'))
model.add(Activation('relu'))

model.add(Convolution2D(nb_filters*4, kernel_size[0], kernel_size[1], border_mode='same'))
model.add(Activation('relu'))

model.add(Convolution2D(nb_filters*4, kernel_size[0], kernel_size[1], border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Flatten())
model.add(Dense(2048))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adamax',  # alternativa: rmsprop?
              metrics=['accuracy'])

model.summary()

####################################################################################################################
# Start training
####################################################################################################################
print("Training Started...")
history = model.fit_generator(trn_dataset,
                              nb_epoch=nb_epoch,
                              samples_per_epoch=trn_samples,
                              validation_data=val_dataset,
                              nb_val_samples=val_samples)
model.save('model.hdf5')

#model = load_model('model.h5')
# Aca hay que elegir los mejores hiperparametros, reentrenar con train+validación (hay que hacer otro generador...no se si hace falta dado que son muchos datos), y despues se evalua en test

print("Evaluating Model...")
scoreT = model.evaluate_generator(tst_dataset, tst_samples)
print("Test score: " + str(scoreT[0]) + "\nTest accuracy: " + str(scoreT[1]))


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

####################################################################################################################
# Confusion matrix
####################################################################################################################
plt.clf()
print("Building Confusion Matrix...")
y_true = []
y_pred = []
i = 0
while i < (int(tst_samples/batch_size) + 1):
    batch = tst_dataset.next()
    y_true.append(np.apply_along_axis(np.argmax, 1, batch[1]))
    y_pred.append(np.apply_along_axis(np.argmax, 1, model.predict(batch[0], batch_size=batch_size)))
    i += 1

y_true = [i for subl in y_true for i in subl]
y_pred = [i for subl in y_pred for i in subl]

classes = np.unique(np.concatenate((y_true, y_pred))) + 32

cnf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(80, 80))
plot_confusion_matrix(cnf_matrix, classes=classes,
                      title='Confusion matrix')
plt.savefig('cnf_matrix.png', format='png')

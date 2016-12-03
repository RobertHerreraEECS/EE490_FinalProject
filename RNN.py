
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.cross_validation import train_test_split
from labelConv import int2label


import pandas as pd
import numpy as np
from sklearn import cross_validation

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from urllib import urlretrieve
import gzip
import sys

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# long short term memory recurrent neural network
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import SimpleRNN
from keras.initializations import normal, identity
from keras.optimizers import RMSprop
from keras.optimizers import Adamax
from keras.utils import np_utils
import string

img_rows, img_cols = 32, 32

# Path of data files
path = "../data"

def convert_(Y):

    alpha = string.letters
    dig   = string.digits
    alphaList = []
    for elem in (alpha + dig):
        alphaList.append(elem)

    list_ = []
    for elem in Y:
        for i in range(0,elem.shape[0]):
            if elem[i] == 1:
                list_.append(i)
    list_ = np.asarray(list_)
    return list_


# Load the preprocessed data and labels
X_train_all = np.load(path+"/trainPreproc_"+str(img_rows)+"_"+str(img_cols)+".npy")
Y_train_all = np.load(path+"/labelsPreproc.npy")

X_train, X_val, Y_train, Y_val = \
    train_test_split(X_train_all, Y_train_all, test_size=0.10, stratify=np.argmax(Y_train_all, axis=1))

print X_train.shape

#

#
print 'Training'

batch_size = 10 #32
nb_classes = 62
nb_epochs = 100
# hidden_units = 100
hidden_units = 1000

learning_rate = 1e-6
clip_norm = 1.0

X_train = X_train.reshape(X_train.shape[0], -1, 1)
X_val = X_val.reshape(X_val.shape[0], -1, 1)
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_train /= 255
X_val /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_val.shape[0], 'test samples')

model = Sequential()
model.add(SimpleRNN(output_dim=hidden_units,
                    init=lambda shape, name: normal(shape, scale=0.01, name=name),
                    inner_init=lambda shape, name: identity(shape, scale=1.0, name=name),
                    activation='sigmoid',
                    input_shape=X_train.shape[1:]))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
rmsprop = RMSprop(lr=learning_rate)
adamax = Adamax(lr=learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=adamax,
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
          verbose=1)#, validation_data=(X_val, Y_val))
print 'Saving model...'
model.save('mymodel.h5') ## this worked, 'mymodel.h5' file is saved to local

# save model

# scores = model.evaluate(X_val, Y_val, verbose=0)
# print('IRNN test score:', scores[0])
# print('IRNN test accuracy:', scores[1])


import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.cross_validation import train_test_split

import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from nolearn.dbn import DBN
import cPickle as pickle


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from urllib import urlretrieve
import gzip
import sys


import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.updates import momentum
from lasagne.updates import sgd
from lasagne.updates import adam
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import string
import argparse


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


def arg_parse():
	'''
		@Description - Reads in terminal input as program input parameters
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_data', action='store', dest='input_data',help='Absolute path to MNIST dataset or arbitrary training data.',required=True)
	parser.add_argument('--params', action='store', dest='model_param_path',help='Name of model files before *_paramfile.w and *_weightfile.w.',required=True)
	parser.add_argument('--version', action='version', version='%(prog)s 1.0')
	parser.add_argument('-v', action='version', version='%(prog)s 1.0')

	results = parser.parse_args()
	return results.input_data,results.model_param_path


def main(input_file, model_path):
    batch_size = 128
    nb_classes = 62 # A-Z, a-z and 0-9
    nb_epoch = 2

    # Input image dimensions
    img_rows, img_cols = 32, 32

    # Path of data files
    path = input_file
    ### PREDICTION ###

    # # Load the model with the highest validation accuracy
    # model.load_weights("best.kerasModelWeights")

    # Load Kaggle test set
    X_test = np.load(path+"/testPreproc_"+str(img_rows)+"_"+str(img_cols)+".npy")


    print X_test.shape

    # Load the preprocessed data and labels
    X_train_all = np.load(path+"/trainPreproc_"+str(img_rows)+"_"+str(img_cols)+".npy")
    Y_train_all = np.load(path+"/labelsPreproc.npy")

    X_train, X_val, Y_train, Y_val = \
        train_test_split(X_train_all, Y_train_all, test_size=0.25, stratify=np.argmax(Y_train_all, axis=1))


    print X_train.shape

    Y_val = convert_(Y_val)


    X_train = X_train.reshape((-1, 1, 32, 32))
    #
    # # input shape for neural network

    # labels = labels.astype(np.uint8)


    X_val = X_val.reshape((-1, 1, 32, 32))
    #
    # # input shape for neural network

    Y_val = Y_val.astype(np.uint8)
    #
    input_image_vector_shape = (None, 1, 32, 32)
    #


    '''
        @description: Two layer convolutional neural network
    '''
    #input layer
    input_layer = ('input', layers.InputLayer)
    # fist layer design
    first_layer_conv_filter = layers.Conv2DLayer
    first_layer_pool_filter = layers.MaxPool2DLayer

    conv_filter = ('conv2d1', first_layer_conv_filter)
    pool_filter = ('maxpool1', first_layer_pool_filter)

    # second layer design
    second_layer_conv_filter = layers.Conv2DLayer
    second_layer_pool_filter = layers.MaxPool2DLayer

    conv_filter2 = ('conv2d2', second_layer_conv_filter)
    pool_filter2 = ('maxpool2', second_layer_pool_filter)

    # dropout rates ( used for regularization )
    dropout_layer = layers.DropoutLayer
    drop1 = 0.5
    drop2 = 0.5
    first_drop_layer = ('dropout1', dropout_layer)
    second_drop_layer = ('dropout2', dropout_layer)
    #
    # network parameters
    design_layers=[input_layer,conv_filter,pool_filter,conv_filter2,pool_filter2,first_drop_layer,('dense', layers.DenseLayer),second_drop_layer,('output', layers.DenseLayer)]
    # Neural net object instance
    net1 = NeuralNet(
        # declare convolutional neural network layers
        # convolutional mapping and pooling window sized will be declared
        # and set to various sizes
        layers=design_layers,
        # input layer
        # vector size of image will be taken as 28 x 28
        input_shape=input_image_vector_shape,
        # first layer convolutional filter
        # mapping layer set at 5 x 5
        conv2d1_num_filters=32,
        conv2d1_filter_size=(5, 5),
        conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d1_W=lasagne.init.HeNormal(gain='relu'),
        # first layer convolutional pool filter
        # mapping layer set at 2 x 2
        maxpool1_pool_size=(2, 2),
        # second layer convolutional filter
        # mapping layer set at 5 x 5
        conv2d2_num_filters=32,
        conv2d2_filter_size=(5, 5),
        conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
        # second layer convolutional pool filter
        # mapping layer set at 2 x 2
        maxpool2_pool_size=(2, 2),
        dropout1_p=drop1,
        # hidden unit density
        dense_num_units=512,
        dense_nonlinearity=lasagne.nonlinearities.rectify,
        # dropout2
        dropout2_p=drop2,
        # output
        output_nonlinearity=lasagne.nonlinearities.softmax,
        #corresponds to the amount of target labels to compare to
        output_num_units=62,
        # optimization method params
        # NOTE: Different momentum steepest gradient methods yield varied
        #       results.
        update=nesterov_momentum,
        # 69
        update_learning_rate=0.01,
        update_momentum=0.078,
        # update_learning_rate=1e-4,
        # update_momentum=0.9,
        # max_epochs=1000,
        # update_learning_rate=0.1,
        # update_momentum=0.003,
        max_epochs=1000,
        verbose=1,
        )
    print "Loading Neural Net Parameters..."
    net1.initialize_layers()
    net1.load_weights_from('{}_weightfile.w'.format(model_path))

    '''
    new_twoLayer_paramfile.w	new_twoLayer_weightfile.w
    '''
    net1.load_params_from('{}_paramfile.w'.format(model_path))


    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

    print 'Testing...'
    y_true, y_pred = Y_val, net1.predict(X_val) # Get our predictions
    print(classification_report(y_true, y_pred)) # Classification on each digit


    print net1.predict(X_val)
    print Y_val
    a = confusion_matrix(Y_val, net1.predict(X_val))
    b = np.trace(a)
    print 'Training Accuracy: ' + str(float(b)/float(np.sum(a)))


if __name__ == '__main__':
    input_file, model = arg_parse()
    main(input_file, model)

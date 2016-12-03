
import os
import argparse
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
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
	parser.add_argument('--output_params', action='store', dest='model_param_path',help='title of output file where model parameters will be stored.\n Will be located in root directory.',required=True)
	parser.add_argument('--version', action='version', version='%(prog)s 1.0')
	parser.add_argument('-v', action='version', version='%(prog)s 1.0')

	results = parser.parse_args()
	return results.input_data,results.model_param_path


def main(file_path, model_path):

    batch_size = 128
    nb_classes = 62 # A-Z, a-z and 0-9
    nb_epoch = 2

    # Input image dimensions
    img_rows, img_cols = 32, 32
    # Path of data files
    # point file path to data folder
    path = file_path


    # Load the preprocessed data and labels
    X_train_all = np.load(path+"/trainPreproc_"+str(img_rows)+"_"+str(img_cols)+".npy")
    Y_train_all = np.load(path+"/labelsPreproc.npy")

    X_train, X_val, Y_train, Y_val = \
        train_test_split(X_train_all, Y_train_all, test_size=0.25, stratify=np.argmax(Y_train_all, axis=1))


    #

    #
    print 'Training...'

    #
    # print 'converting vectors...'
    labels = convert_(Y_train)
    # validation = convert_(Y_val)
    #


    # train = train.reshape((train.shape[0],train.shape[1] * train.shape[2]))
    X_train = X_train.reshape((-1, 1, 32, 32))
    #
    # # input shape for neural network

    labels = labels.astype(np.uint8)
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
    convo_net = NeuralNet(
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




    nn = convo_net.fit(X_train, labels)  # Train CNN


    print 'Saving model paramters to {}_weightfile.w and {}_paramfile.w'.format(model_path,model_path)
    nn.save_weights_to('{}_paramfile.w'.format(model_path))
    nn.save_params_to('{}_weightfile.w'.format(model_path))

T
if __name__ == '__main__':
    file_path, model = arg_parse()
    main(file_path,model)

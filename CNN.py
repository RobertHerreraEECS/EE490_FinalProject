
import os
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
	parser.add_argument('--output_params', action='store', dest='model_param_path',help='title of output file where model parameters will be stored.\n Will be located in root directory.',required=True)
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
        @description: Three layer convolutional neural network
    '''
    # Create Convolutional Neural Net, you may, should and must :) change this attributes to learn how a CNN works
    net1 = NeuralNet(
        layers=[('input', layers.InputLayer),
                ('conv2d1', layers.Conv2DLayer),
                ('maxpool1', layers.MaxPool2DLayer),
                ('conv2d2', layers.Conv2DLayer),
                ('maxpool2', layers.MaxPool2DLayer),
                ('conv2d3', layers.Conv2DLayer),
                ('maxpool3', layers.MaxPool2DLayer),
                # ('conv2d4', layers.Conv2DLayer),
                # ('maxpool4', layers.MaxPool2DLayer),
                ('dropout1', layers.DropoutLayer),
                ('dropout2', layers.DropoutLayer),
                ('dense', layers.DenseLayer),
                # ('dense2', layers.DenseLayer),
                ('output', layers.DenseLayer),
                ],

        input_shape=input_image_vector_shape,

        conv2d1_num_filters=128,
        conv2d1_filter_size=(3, 3),
        conv2d1_nonlinearity=lasagne.nonlinearities.tanh,
        conv2d1_W=lasagne.init.GlorotUniform(),
        conv2d1_pad=(2, 2),
        maxpool1_pool_size=(2, 2),

        conv2d2_num_filters=256,
        conv2d2_filter_size=(3, 3),
        conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d2_pad=(2, 2),
        maxpool2_pool_size=(2, 2),

        conv2d3_num_filters=512,
        conv2d3_filter_size=(3, 3),
        conv2d3_nonlinearity=lasagne.nonlinearities.rectify,
        conv2d3_pad=(2, 2),
        maxpool3_pool_size=(2, 2),

        # conv2d4_num_filters = 1024,
        # conv2d4_filter_size = (2,2),
        # conv2d4_nonlinearity = lasagne.nonlinearities.rectify,
        #
        # maxpool4_pool_size = (2,2),

        dropout1_p=0.5,

        dropout2_p = 0.5,

        dense_num_units=8192,
        dense_nonlinearity=lasagne.nonlinearities.rectify,

        # dense2_num_units = 16,
        # dense2_nonlinearity = lasagne.nonlinearities.rectify,

        output_nonlinearity=lasagne.nonlinearities.softmax,
        output_num_units=62,

        update=momentum,
        # 75.5 with tanh init dense num = 256%
        update_learning_rate=0.03,
        update_momentum=0.8,
        max_epochs=1000,
        verbose=1,
    )

    nn = net1.fit(X_train, labels)  # Train CNN

    print 'Saving file to {}_weightfile.w and {}_paramfile.w'.format(model_path,model_path)
    nn.save_weights_to('{}_weightfile.w'.format(model_path))
    nn.save_params_to('{}_paramfile.w'.format(model_path))

if __name__ == '__main__':
    input_file, model = arg_parse()
    main(input_file,model)

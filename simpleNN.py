from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix
import string
import climate
import theanets
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
import argparse
import os,sys
import cPickle as pickle
import matplotlib.pyplot as plt



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

    print X_train.shape


    labels = convert_(Y_train)
    validation = convert_(Y_val)

    X_train = X_train.reshape((X_train.shape[0],X_train.shape[2] * X_train.shape[3]))
    X_val = X_val.reshape((X_val.shape[0],X_val.shape[2] * X_val.shape[3]))

    print 'Training...'
    class_input = 62
    climate.enable_default_logging()
    # Build a classifier model with 100 inputs and 10 outputs.
    net = theanets.Classifier(layers=[X_train.shape[1], class_input])


    X_train = X_train.astype('f')
    labels = labels.astype('i')

    X_val =  X_val.astype('f')
    validation = validation.astype('i')

    train = X_train, labels
    valid = X_val, validation

    arg = 'adadelta'
    net.train(train, valid, algo=arg, learning_rate=1e-10, momentum=0.00001,input_noise=0.3, hidden_l1=0.1)

    print 'saving model paramters to {}'.format(model_path)
    with open(model_path,'wb') as fid:
        pickle.dump(net, fid)
    print 'Done.'
    # net.train(train, valid, algo=arg, learning_rate=1e-3, momentum=0.7,validate_every = 1000)
    # Show confusion matrices on the training/validation splits.
    # a = confusion_matrix(labels, net.predict(X_train))
    # b = np.trace(a)
    # print 'Training Accuracy: ' + str(float(b)/float(np.sum(a)))
    # a = confusion_matrix(validation, net.predict(X_val))
    # b = np.trace(a)
    # print 'X_val Accuracy: ' + str(float(b)/float(np.sum(a)))
    # acc = float(b)/float(np.sum(a))

if __name__ == '__main__':
    input_file, model = arg_parse()
    main(input_file, model)

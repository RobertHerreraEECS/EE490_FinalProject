
import os
from sklearn.cross_validation import train_test_split
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
from sklearn.metrics import confusion_matrix
import argparse
import string



def arg_parse():
	'''
		@Description - Reads in terminal input as program input parameters
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_data', action='store', dest='input_data',help='Absolute path to MNIST dataset or arbitrary training data.',required=True)
	parser.add_argument('--version', action='version', version='%(prog)s 1.0')
	parser.add_argument('-v', action='version', version='%(prog)s 1.0')

	results = parser.parse_args()
	return results.input_data


def main(input_file):
    batch_size = 128
    nb_classes = 62 # A-Z, a-z and 0-9
    nb_epoch = 2

    # Input image dimensions
    img_rows, img_cols = 32, 32

    # Path of data files
    path = input_file

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
        train_test_split(X_train_all, Y_train_all, test_size=0.25, stratify=np.argmax(Y_train_all, axis=1))

    print X_train.shape


    labels = convert_(Y_train)
    validation = convert_(Y_val)

    X_train = X_train.reshape((X_train.shape[0],X_train.shape[2] * X_train.shape[3]))
    X_val = X_val.reshape((X_val.shape[0],X_val.shape[2] * X_val.shape[3]))

    print 'Training and Testing...'
    clf_rf = RandomForestClassifier()
    clf_rf.fit(X_train, labels)
    y_pred_rf = clf_rf.predict(X_val)
    acD_rf = accuracy_score(validation, y_pred_rf)
    print "random forest accuracy: ",acD_rf


    clf_sgd = SGDClassifier()
    clf_sgd.fit(X_train, labels)
    y_pred_sgd = clf_sgd.predict(X_val)
    acD_sgd = accuracy_score(validation, y_pred_sgd)
    print "stochastic gradient descent accuracy: ",acD_sgd

    clf_svm = LinearSVC()
    clf_svm.fit(X_train, labels)
    y_pred_svm = clf_svm.predict(X_val)
    acD_svm = accuracy_score(validation, y_pred_svm)
    print "Linear SVM accuracy: ",acD_svm

    clf_knn = KNeighborsClassifier()
    clf_knn.fit(X_train, labels)
    y_pred_knn = clf_knn.predict(X_val)
    acD_knn = accuracy_score(validation, y_pred_knn)
    print "nearest neighbors accuracy: ",acD_knn


    clf_nn = DBN([X_train.shape[1], 300, 62],learn_rates=0.0240,learn_rate_decays=0.9,epochs=130)
    clf_nn.fit(X_train, labels)
    acD_nn = clf_nn.score(X_val,validation)
    print "neural network accuracy: ",acD_nn

    clf = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    clf.fit(X_train, labels)
    acD_nn = clf.score(X_val,validation)
    print "naive bayes: ",acD_nn

    clf = BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
    clf.fit(X_train, labels)
    acD_nn = clf.score(X_val,validation)
    print "bernulli naive bayes: ",acD_nn







if __name__ == '__main__':
    input_file = arg_parse()
    main(input_file)

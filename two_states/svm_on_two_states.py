#!/usr/bin/env python

"""
Build SVM learner to predict the molecular photostability based on the first two excited states
"""
import numpy as np
from numpy import genfromtxt
from sklearn import linear_model,svm
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

# load data
def loaddata(filename):
    data = genfromtxt(filename, delimiter=',')
    X = data[1:, 1:5]
    Y = data[1:, -2]
    return X, Y

# grid search on svm parameters
def svm_search(X, Y, params):
    #params = {'C':[0.01,0.05,0.1,0.5,1,5,10,50,100], 'gamma':[0.01,0.05,0.1,0.5,1,5,10,50,100]}
    svc = svm.SVC(kernel='rbf', random_state=0)
    clf = GridSearchCV(svc, params, scoring='f1')
    clf.fit(X,Y)
    return clf.best_estimator_

#calculate performance scores
def calc_score(Y, Z):
    acc, pre, rec = accuracy_score(Y, Z), precision_score(Y, Z),recall_score(Y, Z)
    return acc, pre, rec

def main():
    X, Y = loaddata('train_two_target_states')
    X2, Y2 = loaddata('test_two_target_states')
    avg = np.mean(X, axis=0)
    dev = np.std(X, axis=0)
    Xtrain = (X - avg)/dev
    Xtest = (X2 - avg)/dev
    params = {'C':[0.01,0.05,0.1,0.5,1,5,10,50,100], 'gamma':[0.01,0.05,0.1,0.5,1,5,10,50,100]}
    svmreg = svm_search(Xtrain, Y, params)
    svmreg.fit(Xtrain, Y)
    Z1 = svmreg.predict(Xtrain)
    Z2 = svmreg.predict(Xtest)
    train_acc, train_pre, train_rec = calc_score(Y, Z1)
    test_acc, test_pre, test_rec = calc_score(Y2, Z2)
    print 'training set: accuracy %10.2f, precision %10.2f, recall %10.2f' % (train_acc,train_pre,train_rec)
    print 'test set: accuracy %10.2f, precision %10.2f, recall %10.2f' % (test_acc, test_pre,test_rec)
    print X2[np.where(Y2 != Z2)]

if __name__ == '__main__':
    main()

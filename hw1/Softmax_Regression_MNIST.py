#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import struct
import sys

from utility import load_mnist_images, load_mnist_labels, softmax, \
one_hot_encoding, init_parameters, create_batch, strictly_increasing

'''
print(train_X.shape)    # n_feature * m_train
print(test_X.shape) # n_feature * m_test
print(train_Y.shape)    # 10 * m_train
print(test_Y.shape) # 10 * m_test


print(w.shape)  # 10 * n_feature
print(b.shape)  # 10 * 1
'''
def compute_cost(X, Y, w, b, lambd, regularized):
    m = Y.shape[-1]
    Z = np.dot(w, X) + b
    A = softmax(Z)
    cost = - np.sum(Y * np.log(A)) / m

    if regularized == 1:
        cost += lambd * np.linalg.norm(w, 1) / m # L1 Regularization
    elif regularized == 2:
        cost += lambd * np.linalg.norm(w) / m  # L2 Regularization

    return cost

def softmax_propagate(X, Y, w, b, lambd, regularized):
    m = Y.shape[-1]
    Z = np.dot(w, X) + b
    A = softmax(Z)

    cost = compute_cost(X, Y, w, b, lambd, regularized)

    dw = np.dot((A - Y), X.T) / m + 2 * lambd * w / m
    db = np.sum((A - Y), axis=1, keepdims=True) / m

    if regularized == 2:
        dw += 2 * lambd * w / m
    elif regularized == 1:
        dw += lambd * np.sign(w) / m

    grad = {'dw':dw, 'db':db}

    return grad, cost

def init_adam(w, b):
    v = {}
    s = {}

    v['w'] = np.zeros((w.shape))
    v['b'] = np.zeros((b.shape))

    s['w'] = np.zeros((w.shape))
    s['b'] = np.zeros((b.shape))

    return v, s

def adam_optimize(X, Y, parameters, learning_rate=0.001, lambd=0.01, beta1=0.9, beta2=0.999, epsilon=10**(-8), regularized=2):
    w = parameters['w']
    b = parameters['b']
    v = parameters['v']
    s = parameters['s']

    grad, cost = softmax_propagate(X, Y, w, b, lambd, regularized)

    dw = grad['dw']
    db = grad['db']

    v['w'] = beta1 * v['w'] + (1 - beta1) * dw
    v['b'] = beta1 * v['b'] + (1 - beta1) * db

    s['w'] = beta2 * s['w'] + (1 - beta2) * np.square(dw)
    s['b'] = beta2 * s['b'] + (1 - beta2) * np.square(db)

    w -= learning_rate * (v['w'] / (np.sqrt(s['w']) + epsilon))
    b -= learning_rate * (v['b'] / (np.sqrt(s['b']) + epsilon))

    parameters['w'] = w
    parameters['b'] = b
    parameters['v'] = v
    parameters['s'] = s

    return parameters, cost


class SoftmaxRegression(object):

    def __init__(self, n_feature, n_classes, n_epoch, batch_size=32, learning_rate=0.001,\
     lambd=0.01, beta1=0.9, beta2=0.999, epsilon=10**(-8), regularized=2, T=0.,\
     print_cost=False, print_period=20, record=False, record_period=20, early_stop=False, stop_step=3):
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambd = lambd
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.cost = -1.
        self.regularized = regularized
        self.T = T

        w, b = init_parameters(n_feature, n_classes)
        v, s = init_adam(w, b)
        self.parameters = {'w':w, 'b':b, 'v':v, 's':s}

        self.print_cost = print_cost
        self.print_period = print_period
        self.record = record
        self.record_period = record_period
        self.early_stop = early_stop
        self.stop_step = stop_step

    def fit(self, X, Y, holdout_X=None, holdout_Y=None, test_X=None, test_Y=None):
        train = {'cost': [], 'accuracy': []}
        val = {'cost': [], 'accuracy':[]}
        test = {'cost':[], 'accuracy':[]}

        for i in range(self.n_epoch):
            X_batches, Y_batches, n_batch = create_batch(X, Y, self.batch_size)

            for j in range(n_batch):
                self.parameters, self.cost = adam_optimize(X_batches[j], Y_batches[j], self.parameters, \
                self.learning_rate, self.lambd, self.beta1, self.beta2, self.epsilon, self.regularized)

            self.learning_rate /= 1. + self.T

            # record plot data, toggle by set record to True and set record period
            if self.record and (i+1) % self.record_period == 0:
                trainAcc = self.predict(X, Y)
                train['cost'].append(self.cost)
                train['accuracy'].append(trainAcc)

                if holdout_X is not None:
                    valCost = compute_cost(holdout_X, holdout_Y, self.parameters['w'], self.parameters['b'], self.lambd, self.regularized)
                    valAcc = self.predict(holdout_X, holdout_Y)
                    val['cost'].append(valCost)
                    val['accuracy'].append(valAcc)

                    if self.early_stop and (i+1) / self.record_period > self.stop_step and strictly_increasing(val['cost'][-self.stop_step:]):
                        print('Early stop at %d epoch' % (i+1))
                        break

                if test_X is not None:
                    testCost = compute_cost(test_X, test_Y, self.parameters['w'], self.parameters['b'], self.lambd, self.regularized)
                    testAcc = self.predict(test_X, test_Y)
                    test['cost'].append(testCost)
                    test['accuracy'].append(testAcc)

            # print cost, toggle by set print_cost to True and set print period
            if self.print_cost and (i+1) % self.print_period == 0:
                print('%d epoches cost: %f' % (i+1, self.cost))

        return train, val, test


    def predict(self, X, Y):
        m = X.shape[-1]

        w = self.parameters['w']
        b = self.parameters['b']
        Z = np.dot(w, X) + b
        A = softmax(Z)

        self.Y_p = np.argmax(A, axis=0)
        Y_label = np.argmax(Y, axis=0)
        correct = (self.Y_p == Y_label)

        self.accuracy = np.sum(correct)/ m

        return self.accuracy

  #!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import struct
import sys

from utility import load_mnist_images, load_mnist_labels, sigmoid, \
init_parameters, create_batch, extract_target_data, strictly_increasing


def compute_cost(X, Y, w, b, lambd, regularized):
    m = Y.shape[-1]
    Z = np.dot(w, X) + b
    A = sigmoid(Z)
    cost =  - np.sum(Y * np.log(A) + (1-Y) * np.log(1-A)) / m

    if regularized == 1:
            cost += lambd * np.linalg.norm(w, 1) / m
    elif regularized == 2:
            cost += lambd * np.linalg.norm(w, 2) / m

    return cost

def logistic_propagate(X, Y, w, b, lambd, regularized):
    m = Y.shape[-1]
    Z = np.dot(w, X) + b
    A = sigmoid(Z)

    cost = compute_cost(X, Y, w, b, lambd, regularized)

    dw = np.dot((A - Y), X.T) / m
    db = np.sum((A - Y), axis=1, keepdims=True) / m

    if regularized == 2:
        dw += 2 * lambd * w / m
    elif regularized == 1:
        dw += lambd * np.sign(w) / m

    grad = {'dw':dw, 'db':db}

    return grad, cost


def optimize(X, Y, w, b, learning_rate, lambd, regularized):
    grad, cost = logistic_propagate(X, Y, w, b, lambd, regularized)

    dw = grad['dw']
    db = grad['db']

    w -= learning_rate * dw
    b -= learning_rate * db

    return w, b, cost


class LogisticRegression(object):

    def __init__(self, n_feature, n_epoch, batch_size=32, learning_rate=0.001, lambd=0.01,\
     regularized=2,  T=0., print_cost=False, print_period=20, record=False, record_period=20, early_stop=False, stop_step=3):
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambd = lambd
        self.w, self.b = init_parameters(n_feature, 1)
        self.cost = -1.
        self.regularized = regularized
        self.T = T

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
                self.w, self.b, self.cost = optimize(X_batches[j], Y_batches[j], self.w, self.b, self.learning_rate, self.lambd, self.regularized)

            self.learning_rate /= 1. + self.T

            # record plot data, toggle by set record to True and set record period
            if self.record and (i+1) % self.record_period == 0:
                trainAcc = self.predict(X, Y)
                train['cost'].append(self.cost)
                train['accuracy'].append(trainAcc)

                if holdout_X is not None:
                    valCost = compute_cost(holdout_X, holdout_Y, self.w, self.b, self.lambd, self.regularized)
                    valAcc = self.predict(holdout_X, holdout_Y)
                    val['cost'].append(valCost)
                    val['accuracy'].append(valAcc)

                    if self.early_stop and (i+1) / self.record_period > self.stop_step and strictly_increasing(val['cost'][-self.stop_step:]):
                        print('Early stop at %d epoch' % (i+1))
                        break

                if test_X is not None:
                    testCost = compute_cost(test_X, test_Y, self.w, self.b, self.lambd, self.regularized)
                    testAcc = self.predict(test_X, test_Y)
                    test['cost'].append(testCost)
                    test['accuracy'].append(testAcc)

            # print cost, toggle by set print_cost to True and set print period
            if self.print_cost and (i+1) % self.print_period == 0:
                print('%d epoches cost: %f' % (i+1, self.cost))

        return train, val, test


    def predict(self, X, Y):
        m = X.shape[-1]

        Z = np.dot(self.w, X) + self.b
        A = sigmoid(Z)

        self.Y_p = (A > 0.5)
        correct = (self.Y_p == Y)
        self.accuracy = np.sum(correct) / m
        return self.accuracy

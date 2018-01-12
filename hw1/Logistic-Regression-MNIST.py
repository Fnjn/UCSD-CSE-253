#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import struct
import sys

from utility import load_mnist_images, load_mnist_labels, sigmoid, \
init_parameters, create_batch


def extract_target_data(X, Y, target1, target2):
    p1 = (Y == target1)
    p2 = (Y == target2)
    p = p1 | p2
    x_target = X[:,p]
    y_target = Y[p]
    y_target = (y_target == target1)
    y_target = np.expand_dims(y_target, axis=0)
    return x_target, y_target

def compute_cost(Y, A, m, w, lambd, regularized=0):
    cost =  - np.sum(Y * np.log(A) + (1-Y) * np.log(1-A)) / m
    if regularized == 1:
            cost += lambd * np.linalg.norm(w, 1)
    elif regularized == 2:
            cost += lambd * np.linalg.norm(w) / m

    return cost


def logistic_propagate(X, Y, w, b, lambd):
    m = Y.shape[-1]
    Z = np.dot(w, X) + b
    A = sigmoid(Z)

    cost = compute_cost(Y, A, m, w, lambd, 2)

    dw = np.dot((A - Y), X.T) / m + 2 * lambd * w / m
    db = np.sum((A - Y), axis=1, keepdims=True) / m

    grad = {'dw':dw, 'db':db}

    return grad, cost


def optimize(X, Y, w, b, learning_rate, lambd):
    grad, cost = logistic_propagate(X, Y, w, b, lambd)

    dw = grad['dw']
    db = grad['db']

    w -= learning_rate * dw
    b -= learning_rate * db

    return w, b, cost


class LogisticRegression(object):

    def __init__(self, n_feature, n_epoch, batch_size=32, learning_rate=0.001, lambd=0.01):
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambd = lambd
        self.w, self.b = init_parameters(n_feature, 1)
        self.cost = -1.

    def fit(self, X, Y):
        train_cost = []
        val_cost = []

        for i in range(self.n_epoch):
            X_batches, Y_batches, n_batch = create_batch(X, Y, self.batch_size)
            holdout = max(1,n_batch/10)
            for j in range(holdout, n_batch):
                self.w, self.b, self.cost = optimize(X_batches[j], Y_batches[j], self.w, self.b, self.learning_rate, self.lambd)

            train_cost.append(self.cost)

            holdoutX = np.concatenate(X_batches[:holdout], axis=1)
            holdoutY = np.concatenate(np.concatenate(Y_batches[:holdout], axis=0))
            m = holdoutY.shape[-1]
            Z = np.dot(self.w, holdoutX) + self.b
            A = sigmoid(Z)
            val_cost.append(compute_cost(holdoutY, A, m, self.w, self.lambd, 2))

            if i % 20 == 0: print('%d epoches cost: %f' % (i, self.cost))
        return train_cost, val_cost

    def predict(self, X, Y):
        m = X.shape[-1]

        Z = np.dot(self.w, X) + self.b
        A = sigmoid(Z)

        self.Y_p = (A > 0.5)
        correct = (self.Y_p == Y)
        self.accuracy = np.sum(correct)*1.0 / m


def __main__():
    train_images = load_mnist_images('train-images.idx3-ubyte', 20000)
    train_labels = load_mnist_labels('train-labels.idx1-ubyte', 20000)
    test_images = load_mnist_images('t10k-images.idx3-ubyte')
    test_labels = load_mnist_labels('t10k-labels.idx1-ubyte')

    test_images = test_images[-2000:]
    test_labels = test_labels[-2000:]

    '''
    # Show A Image
    plt.gray()
    plt.imshow(train_images[50])
    plt.show()
    '''

    m_train = train_images.shape[0]
    m_test = test_images.shape[0]
    train_X = train_images.reshape(m_train, -1).T / 255.
    test_X = test_images.reshape(m_test, -1).T / 255.

    train_X, train_Y = extract_target_data(train_X, train_labels, 2, 3)
    test_X, test_Y = extract_target_data(test_X, test_labels, 2, 3)

    n_feature = train_X.shape[0]

    sigmoid_2_model = LogisticRegression(n_feature, n_epoch=400)
    train_cost, val_cost = sigmoid_2_model.fit(train_X, train_Y)
    sigmoid_2_model.predict(test_X, test_Y)
    print('Softmax Regression on Category 2 and 3 Accuracy: %f %%' % (sigmoid_2_model.accuracy * 100))

    tc_plt, = plt.plot(train_cost, label='Training Cost')
    vc_plt, = plt.plot(val_cost, label='Validating Cost')
    plt.legend(handles=[tc_plt, vc_plt])
    plt.show()
__main__()

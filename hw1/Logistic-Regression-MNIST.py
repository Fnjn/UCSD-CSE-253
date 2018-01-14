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

def compute_cost(X, Y, w, b, lambd, regularized=0):
    m = Y.shape[-1]
    Z = np.dot(w, X) + b
    A = sigmoid(Z)
    cost =  - np.sum(Y * np.log(A) + (1-Y) * np.log(1-A)) / m

    if regularized == 1:
            cost += lambd * np.linalg.norm(w, 1) / m
    elif regularized == 2:
            cost += lambd * np.linalg.norm(w, 2)

    return cost


def logistic_propagate(X, Y, w, b, lambd):
    m = Y.shape[-1]
    Z = np.dot(w, X) + b
    A = sigmoid(Z)

    cost = compute_cost(X, Y, w, b, lambd, 2)

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


    def fit(self, X, Y, holdout=0.1, test_X=None, test_Y=None):
        train = {'cost': [], 'accuracy': []}
        val = {'cost': [], 'accuracy':[]}
        test = {'cost':[], 'accuracy':[]}

        m = X.shape[-1]
        permutation = np.random.permutation(m)
        X_shuffle = X[:, permutation]
        Y_shuffle = Y[:, permutation]
        m_holdout = int(m*holdout)


        for i in range(self.n_epoch):
            X_batches, Y_batches, n_batch = create_batch(X[:, m_holdout:], Y[:, m_holdout:], self.batch_size)

            for j in range(n_batch):
                self.w, self.b, self.cost = optimize(X_batches[j], Y_batches[j], self.w, self.b, self.learning_rate, self.lambd)

            if i % 20 == 0:
                print('%d epoches cost: %f' % (i, self.cost))

                # Recording data for plotting
                train['cost'].append(self.cost)
                val['cost'].append(compute_cost(X[:, :m_holdout], Y[:, :m_holdout], self.w, self.b, self.lambd, 2))
                train['accuracy'].append(self.predict(X[:, m_holdout:], Y[:, m_holdout:]))
                val['accuracy'].append(self.predict(X[:, :m_holdout], Y[:, :m_holdout]))
                if test_X is not None and test_Y is not None:
                    test['cost'].append(compute_cost(test_X, test_Y, self.w, self.b, self.lambd, 2))
                    test['accuracy'].append(self.predict(test_X, test_Y))



        return train, val, test

    def predict(self, X, Y):
        m = X.shape[-1]

        Z = np.dot(self.w, X) + self.b
        A = sigmoid(Z)

        self.Y_p = (A > 0.5)
        correct = (self.Y_p == Y)
        self.accuracy = np.sum(correct) / m
        return self.accuracy


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
    train, val, test = sigmoid_2_model.fit(train_X, train_Y, test_X=test_X, test_Y=test_Y)
    sigmoid_2_model.predict(test_X, test_Y)
    print('Softmax Regression on Category 2 and 3 Accuracy: %f %%' % (sigmoid_2_model.accuracy * 100))

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(211)
    tc_plt, = ax1.plot(train['cost'], label='Training Cost')
    vc_plt, = ax1.plot(val['cost'], label='Validation Cost')
    tec_plt, =  ax1.plot(test['cost'], label='Test Cost')
    ax1.legend(handles=[tc_plt, vc_plt, tec_plt])
    ax2 = fig1.add_subplot(212)
    ta_plt, = ax2.plot(train['accuracy'], label='Training Accuracy')
    va_plt, = ax2.plot(val['accuracy'], label='Validating Accuracy')
    tea_plt, =  ax2.plot(test['accuracy'], label='Test Accuracy')
    ax2.legend(handles=[ta_plt, va_plt, tea_plt], loc=4)
    plt.show()


    fig2 = plt.figure()
    ax3 = fig2.add_subplot(111)
    ax3.imshow(sigmoid_2_model.w.reshape(28,28))
    plt.show()

__main__()

#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import struct
import sys

def load_mnist_images(path, max_images=sys.maxsize):
    with open(path, 'rb') as f:
        f.read(4)
        n = int(struct.unpack('>i', f.read(4))[0])
        n = min(max_images, n)
        n_rows = int(struct.unpack('>I', f.read(4))[0])
        n_cols = int(struct.unpack('>I', f.read(4))[0])
        images = []
        for i in range(n):
            image = np.zeros((n_rows,n_cols), dtype=np.int16)
            for r in range(n_rows):
                for c in range(n_cols):
                    image[r,c] = int(struct.unpack('>B', f.read(1))[0])
            images.append(image)
    return np.array(images)

def load_mnist_labels(path, max_labels=sys.maxsize):
    with open(path, 'rb') as f:
        f.read(4)
        n = int(struct.unpack('>i', f.read(4))[0])
        n = min(max_labels, n)
        labels = []
        for i in range(n):
            label = int(struct.unpack('>B', f.read(1))[0])
            labels.append(label)
    return np.array(labels)

def one_hot_encoding(labels, n_feature):
    m = labels.shape[0]
    encode = np.zeros((n_feature, m), dtype=int)
    for i in range(m):
        encode[labels[i], i] = 1
    return encode

'''
print(train_X.shape)    # n_feature * m_train
print(test_X.shape) # n_feature * m_test
print(train_Y.shape)    # 10 * m_train
print(test_Y.shape) # 10 * m_test
'''

def init_parameters(dim1, dim2):
    w = np.zeros((dim2, dim1))
    #w = np.random.randn(dim2, dim1) * 0.01 + 0.5
    b = np.zeros((dim2, 1))
    return w, b

'''
n_feature = train_X.shape[0]    # 784 (28 * 28)
n_output = 10

w, b = init_parameters(n_feature, n_output)

print(w.shape)  # 10 * n_feature
print(b.shape)  # 10 * 1
'''

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def softmax(x):
    z = np.exp(x)
    z = z / np.sum(z, axis=0, keepdims=True)
    return z

def logistic_propagate(X, Y, w, b, lambd):
    m = Y.shape[-1]
    Z = np.dot(w, X) + b
    A = sigmoid(Z)

    cost = - np.sum(Y * np.log(A) + (1-Y) * np.log(1-A)) / m

    cost += lambd * np.linalg.norm(w) / m  # L2 Regularization
    #cost += lambd * np.linalg.norm(w, 1)    # L1 Regularization

    dw = np.dot((A - Y), X.T) / m + 2 * lambd * w / m
    db = np.sum((A - Y), axis=1, keepdims=True) / m

    grad = {'dw':dw, 'db':db}

    return grad, cost

def softmax_propagate(X, Y, w, b, lambd):
    m = Y.shape[-1]
    Z = np.dot(w, X) + b
    A = softmax(Z)

    cost = - np.sum(Y * np.log(A)) / m
    cost += lambd * np.linalg.norm(w) / m  # L2 Regularization
    #cost += lambd * np.linalg.norm(w, 1)    # L1 Regularization

    dw = np.dot((A - Y), X.T) / m + 2 * lambd * w / m
    db = np.sum((A - Y), axis=1, keepdims=True) / m

    grad = {'dw':dw, 'db':db}

    return grad, cost

'''
def optimize(X, Y, w, b, learning_rate):
    grad, cost = logistic_propagate(X, Y, w, b)

    dw = grad['dw']
    db = grad['db']

    w -= learning_rate * dw
    b -= learning_rate * db

    return w, b, cost
'''

def init_adam(w, b):
    v = {}
    s = {}

    v['w'] = np.zeros((w.shape))
    v['b'] = np.zeros((b.shape))

    s['w'] = np.zeros((w.shape))
    s['b'] = np.zeros((b.shape))

    return v, s

def adam_optimize(X, Y, w, b, v, s, regression='logistic', learning_rate=0.001, lambd=0.01, beta1=0.9, beta2=0.999, epsilon=10**(-8)):
    if regression == 'logistic':
        grad, cost = logistic_propagate(X, Y, w, b, lambd)
    else:
        grad, cost = softmax_propagate(X, Y, w, b, lambd)

    dw = grad['dw']
    db = grad['db']

    v['w'] = beta1 * v['w'] + (1 - beta1) * dw
    v['b'] = beta1 * v['b'] + (1 - beta1) * db

    s['w'] = beta2 * s['w'] + (1 - beta2) * np.square(dw)
    s['b'] = beta2 * s['b'] + (1 - beta2) * np.square(db)

    w -= learning_rate * (v['w'] / (np.sqrt(s['w']) + epsilon))
    b -= learning_rate * (v['b'] / (np.sqrt(s['b']) + epsilon))

    return w, b, v, s, cost


def create_batch(X, Y, batch_size):
    m = X.shape[-1]
    n_batch = int(m / batch_size)

    X_batches = []
    Y_batches = []

    permutation = np.random.permutation(m)
    X_shuffle = X[:, permutation]
    Y_shuffle = Y[:, permutation]

    for i in range(n_batch):
        X_batch = X_shuffle[:, i * batch_size: (i+1) * batch_size]
        Y_batch = Y_shuffle[:, i * batch_size: (i+1) * batch_size]
        X_batches.append(X_batch)
        Y_batches.append(Y_batch)

    if m % n_batch != 0:
        X_batch = X_shuffle[:, n_batch * batch_size:]
        Y_batch = Y_shuffle[:, n_batch * batch_size:]
        X_batches.append(X_batch)
        Y_batches.append(Y_batch)
        n_batch += 1

    return X_batches, Y_batches, n_batch

'''
def batch_gradient(X, Y, n_epoch, batch_size, learning_rate):
    m = X.shape[-1]
    n_feature = X.shape[0]
    n_output = 10

    w, b = init_parameters(n_feature, n_output)
    cost = -1.

    for i in range(n_epoch):
        X_batches, Y_batches, n_batch = create_batch(X, Y, batch_size)

        for j in range(n_batch):
            w, b, cost = optimize(X_batches[j], Y_batches[j], w, b, learning_rate)

        if i % 20 == 0: print(i, cost)

    parameters = {'w':w, 'b':b}
    return parameters, cost
'''

def batch_gradient_with_adam(X, Y, n_epoch, batch_size, learning_rate, regression, lambd):
    m = X.shape[-1]
    n_feature = X.shape[0]
    n_output = 10

    w, b = init_parameters(n_feature, n_output)
    v, s = init_adam(w, b)
    cost = -1.

    for i in range(n_epoch):
        X_batches, Y_batches, n_batch = create_batch(X, Y, batch_size)

        for j in range(n_batch):
            w, b, v, s, cost = adam_optimize(X_batches[j], Y_batches[j], w, b, v, s, regression, learning_rate, lambd=lambd)

        if i % 20 == 0: print('%d epoches cost: %f' % (i, cost))

    parameters = {'w':w, 'b':b}
    return parameters, cost


def predict(X, Y, parameters):
    m = X.shape[-1]

    w = parameters['w']
    b = parameters['b']
    Z = np.dot(w, X) + b
    A = sigmoid(Z)

    P = np.argmax(A, axis=0)
    correct = (P == Y)
    accuracy = np.sum(correct) / m

    return P, accuracy


def __main__():
    train_images = load_mnist_images('train-images.idx3-ubyte', 2000)
    train_labels = load_mnist_labels('train-labels.idx1-ubyte', 2000)
    test_images = load_mnist_images('t10k-images.idx3-ubyte', 200)
    test_labels = load_mnist_labels('t10k-labels.idx1-ubyte', 200)

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


    train_Y = one_hot_encoding(train_labels, 10)
    test_Y = one_hot_encoding(test_labels, 10)

    #parameters, cost = batch_gradient(train_X, train_Y, n_epoch=400, batch_size=32, learning_rate=0.001)
    parameters, cost = batch_gradient_with_adam(train_X, train_Y, n_epoch=200, batch_size=32, learning_rate=0.01, regression='logistic', lambd=0.01)
    P, accuracy = predict(test_X, test_labels, parameters)
    print('Logistic Regression Accuracy: %f %%' % (accuracy * 100))

    parameters, cost = batch_gradient_with_adam(train_X, train_Y, n_epoch=200, batch_size=32, learning_rate=0.01, regression='softmax', lambd=0.01)
    P, accuracy = predict(test_X, test_labels, parameters)
    print('Softmax Regression Accuracy: %f %%' % (accuracy * 100))

__main__()

#!/usr/bin/env python3

import numpy as np

from utility import init_parameters, init_adam, sigmoid, softmax, tanh, create_batch

def init_parameters_layers(layers):
    parameters = {}
    for i in range(1, len(layers)):
        w, b = init_parameters(layers[i-1], layers[i])
        v, s = init_adam(w, b)
        parameters['w'+str(i)] = w
        parameters['b'+str(i)] = b
        parameters['v'+str(i)] = v
        parameters['s'+str(i)] = s
    return parameters

def linear_forward_block(a_prev, w, b, activation):
    h = np.dot(w, a_prev) + b
    if activation == 'sigmoid':
        a = sigmoid(h)
    else:
        a = tanh(h)

    cache = {'a_prev':a_prev, 'w':w, 'b':b}

    return a, cache

def softmax_forward_block(a_prev, w, b):
    h = np.dot(w, a_prev) + b
    a = softmax(h)

    cache = {'a_prev':a_prev, 'w':w, 'b':b}

    return a, cache

def linear_backward_block(a, da, cache, activation):
    a_prev = cache['a_prev']
    w = cache['w']
    b = cache['b']

    m = a_prev.shape[1]

    if activation == 'sigmoid':
        dh = a * (1 - a)
    else:
        dh = 1 - (a ** 2)

    dh *= da

    dw = np.dot(dh, a_prev.T)
    db = np.sum(dh, axis=1, keepdims=True)
    da_prev = np.dot(w.T, dh)

    grad = {'dw':dw, 'db':db, 'da_prev':da_prev}

    return grad

def softmax_backward_block(a, y, cache, lambd):
    a_prev = cache['a_prev']
    w = cache['w']
    b = cache['b']

    m = a_prev.shape[1]

    dh = a - y
    dw = np.dot(dh, a_prev.T) / m #+ 2 * lambd * w / m
    db = np.sum(dh, axis=1, keepdims=True) / m
    da_prev = np.dot(w.T, dh)

    grad = {'dw':dw, 'db':db, 'da_prev':da_prev}

    return grad

def forward(x, y, parameters, layers, lambd, predict=False):
    caches = []
    activation_caches = []

    m = x.shape[-1]
    n_layers = len(layers)
    a = x

    for i in range(1, n_layers-1):
        a, cache = linear_forward_block(a, parameters['w'+str(i)], parameters['b'+str(i)],activation='sigmoid')
        activation_caches.append(a)
        caches.append(cache)

    p, cache = softmax_forward_block(a, parameters['w'+str(n_layers-1)], parameters['b'+str(n_layers-1)])
    activation_caches.append(p)
    caches.append(cache)

    if predict:
        return p

    cost = -np.sum(y * np.log(p)) / m

    return activation_caches, caches, cost

def backward(a_caches, caches, y, lambd):
    grads = []

    grad = softmax_backward_block(a_caches[-1], y, caches[-1], lambd)
    grads.append(grad)
    for i in reversed(range(len(a_caches)-1)):
        grad = linear_backward_block(a_caches[i], grads[-1]['da_prev'], caches[i], activation='sigmoid')
        grads.append(grad)

    return list(reversed(grads))

def adam_optimize(X, Y, parameters, layers, learning_rate, lambd, beta1, beta2, epsilon):
    activation_caches, caches, cost = forward(X, Y, parameters, layers, lambd)
    grad = backward(activation_caches, caches, Y, lambd)

    for i in range(1, len(layers)):
        dw = grad[i-1]['dw']
        db = grad[i-1]['db']

        w = parameters['w'+str(i)]
        b = parameters['b'+str(i)]
        v = parameters['v'+str(i)]
        s = parameters['s'+str(i)]

        v['w'] = beta1 * v['w'] + (1 - beta1) * dw
        v['b'] = beta1 * v['b'] + (1 - beta1) * db

        s['w'] = beta2 * s['w'] + (1 - beta2) * np.square(dw)
        s['b'] = beta2 * s['b'] + (1 - beta2) * np.square(db)

        w -= learning_rate * (v['w'] / (np.sqrt(s['w']) + epsilon))
        b -= learning_rate * (v['b'] / (np.sqrt(s['b']) + epsilon))

        parameters['w'+str(i)] = w
        parameters['b'+str(i)] = b
        parameters['v'+str(i)] = v
        parameters['s'+str(i)] = s

    return parameters, cost

class NN_model(object):

    def __init__(self, layers, n_epoch, batch_size=32, learning_rate=0.001, lambd=0.01, beta1=0.9, beta2=0.999, epsilon=10**(-8)):
        self.layers = layers
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambd = lambd
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.parameters = init_parameters_layers(layers)
        self.cost = -1.

    def fit(self, X, Y):
        for i in range(self.n_epoch):
            X_batches, Y_batches, n_batch = create_batch(X, Y, self.batch_size)

            for j in range(n_batch):
                self.parameters, self.cost = adam_optimize(X_batches[j], Y_batches[j], self.parameters, \
                self.layers, self.learning_rate, self.lambd, self.beta1, self.beta2, self.epsilon)

            if i % 20 == 0: print('%d epoches cost: %f' % (i, self.cost))

    def predict(self, X, Y):
        m = X.shape[-1]

        p = forward(X, None, self.parameters, self.layers, self.lambd, predict=True)
        self.Y_p = np.argmax(p, axis=0)
        Y_labels = np.argmax(Y, axis=0)
        correct = (self.Y_p == Y_labels)
        self.accuracy = np.sum(correct) / m

        return self.accuracy

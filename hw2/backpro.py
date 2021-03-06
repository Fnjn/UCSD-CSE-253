#!/usr/bin/env python3

import numpy as np

from utility import init_parameters, init_adam, sigmoid, softmax, tanh,\
 create_batch, grad_to_vec, dict_to_vec, vec_to_dict, strictly_increasing

def init_parameters_layers(layers):
    parameters = {}
    n_input = layers[0]
    for i in range(1, len(layers)):
        # w, b = init_parameters(layers[i-1], layers[i])
        w, b = init_parameters(layers[i-1], layers[i], factor=1./np.sqrt(n_input))
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

def backward(a_caches, caches, y, lambd, activation):
    grads = []

    grad = softmax_backward_block(a_caches[-1], y, caches[-1], lambd)
    grads.append(grad)
    for i in reversed(range(len(a_caches)-1)):
        grad = linear_backward_block(a_caches[i], grads[-1]['da_prev'], caches[i], activation=activation)
        grads.append(grad)

    return list(reversed(grads))

def adam_optimize(X, Y, parameters, layers, learning_rate, lambd, beta1, beta2, epsilon, activation='sigmoid'):
    activation_caches, caches, cost = forward(X, Y, parameters, layers, lambd)
    grad = backward(activation_caches, caches, Y, lambd, activation)

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

    def __init__(self, layers, n_epoch, batch_size=128, learning_rate=0.001, lambd=0.01, \
    beta1=0.9, beta2=0.999, epsilon=10**(-8), activation='sigmoid', print_cost=True, print_period=20):
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
        self.activation = activation

        self.print_cost = print_cost
        self.print_period = print_period


    def fit(self, X, Y, holdout_X, holdout_Y, test_X=None, test_Y=None):
        train = {'cost': [], 'accuracy': []}
        val = {'cost': [], 'accuracy':[]}
        test = {'cost':[], 'accuracy':[]}

        self.min_val_acc = 0.
        self.min_param = None

        for i in range(self.n_epoch):
            X_batches, Y_batches, n_batch = create_batch(X, Y, self.batch_size)

            for j in range(n_batch):
                self.parameters, self.cost = adam_optimize(X_batches[j], Y_batches[j], self.parameters, \
                self.layers, self.learning_rate, self.lambd, self.beta1, self.beta2, self.epsilon)

            trainAcc = self.predict(X, Y)
            train['cost'].append(self.cost)
            train['accuracy'].append(trainAcc)

            _, _, valCost = forward(holdout_X, holdout_Y, self.parameters, self.layers, self.lambd)
            valAcc = self.predict(holdout_X, holdout_Y)
            val['cost'].append(valCost)
            val['accuracy'].append(valAcc)

            if valAcc > self.min_val_acc:
                self.min_param = self.parameters
                self.min_val_acc = valAcc

            if test_X is not None:
                _, _, testCost = forward(test_X, test_Y, self.parameters, self.layers, self.lambd)
                testAcc = self.predict(test_X, test_Y)
                test['cost'].append(testCost)
                test['accuracy'].append(testAcc)

            # print cost, toggle by set print_cost to True and set print period
            if self.print_cost and (i+1) % self.print_period == 0:
                print('%d epoches cost: %f' % (i+1, self.cost))

        self.parameters = self.min_param

        return train, val, test

    def predict(self, X, Y):
        m = X.shape[-1]

        p = forward(X, None, self.parameters, self.layers, self.lambd, predict=True)
        self.Y_p = np.argmax(p, axis=0)
        Y_labels = np.argmax(Y, axis=0)
        correct = (self.Y_p == Y_labels)
        self.accuracy = np.sum(correct) / m

        return self.accuracy

    def gradient_check(self, X, Y):
        n_layers = len(self.layers)
        m = X.shape[-1]
        epsilon= 10e-2

        activation_caches, caches, cost = forward(X, Y, self.parameters, self.layers, self.lambd)
        gradient = backward(activation_caches, caches, Y, self.lambd, self.activation)

        parameters_vec = dict_to_vec(self.parameters, self.layers)
        grad = grad_to_vec(gradient, self.layers)

        perm = np.random.permutation(grad.shape[0])
        grad = grad[perm[:3000]]
        grad_approx = np.zeros((grad.shape))

        for i in range(grad.shape[0]):
            p_plus = np.copy(parameters_vec)
            p_minus = np.copy(parameters_vec)

            p_plus[perm[i]] += epsilon
            p_minus[perm[i]] -= epsilon

            _, _, cost_plus = forward(X, Y, vec_to_dict(p_plus, self.layers), self.layers, self.lambd)
            _, _, cost_minus = forward(X, Y, vec_to_dict(p_minus, self.layers), self.layers, self.lambd)

            grad_approx[i] = (cost_plus - cost_minus) * m

        grad_approx /= 2 * epsilon
        diff = np.abs(grad - grad_approx)
        self.diff = diff

        if any(diff > 10e-4):
            print('Gradient check found %d possible problems.' % (np.sum(diff > 10e-4, dtype=int)))
        else:
            print('No problem found in gradient check.')

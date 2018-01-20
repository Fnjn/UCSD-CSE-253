#!/usr/bin/env python3

import numpy as np
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

def split_holdout(X, Y, holdout_ratio):
    m = X.shape[-1]
    permutation = np.random.permutation(m)
    X_shuffle = X[:, permutation]
    Y_shuffle = Y[:, permutation]
    m_holdout = int(m * holdout_ratio)
    return X[:,:-m_holdout], Y[:,:-m_holdout], X[:,-m_holdout:], Y[:,-m_holdout:]

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def softmax(x):
    z = np.exp(x)
    z = z / np.sum(z, axis=0, keepdims=True)
    return z

def tanh(x):
    return np.tanh(x)

def one_hot_encoding(labels, n_feature):
    m = labels.shape[0]
    encode = np.zeros((n_feature, m), dtype=int)
    for i in range(m):
        encode[labels[i], i] = 1
    return encode

def init_parameters(dim1, dim2, setZero=False):
    if setZero:
        w = np.zeros((dim2, dim1))
    else:
        w = np.random.randn(dim2, dim1) * 0.01

    b = np.zeros((dim2, 1))
    return w, b

def init_adam(w, b):
    v = {}
    s = {}

    v['w'] = np.zeros((w.shape))
    v['b'] = np.zeros((b.shape))

    s['w'] = np.zeros((w.shape))
    s['b'] = np.zeros((b.shape))

    return v, s

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

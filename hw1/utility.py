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

def extract_target_data(X, Y, target1, target2):
    p1 = (Y == target1)
    p2 = (Y == target2)
    p = p1 | p2
    x_target = X[:,p]
    y_target = Y[p]
    y_target = (y_target == target1)
    y_target = np.expand_dims(y_target, axis=0)
    return x_target, y_target

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def softmax(x):
    z = np.exp(x)
    z = z / np.sum(z, axis=0, keepdims=True)
    return z

def one_hot_encoding(labels, n_feature):
    m = labels.shape[0]
    encode = np.zeros((n_feature, m), dtype=int)
    for i in range(m):
        encode[labels[i], i] = 1
    return encode

def init_parameters(dim1, dim2, setZero=True):
    if setZero:
        w = np.zeros((dim2, dim1))
    else:
        w = np.random.randn(dim2, dim1) * 0.01

    b = np.zeros((dim2, 1))
    return w, b

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

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

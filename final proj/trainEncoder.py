#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

with tf.device('/device:GPU:0'):
    x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name='x')
    y = tf.placeholder(tf.int32, shape=[None, ny], name='y')

    encoder = IND_y(x)
    losss = tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=encoder)

    evaluate = tf.equal(tf.round(encoder), y)
    accuracy = tf.reduce_mean(tf.cast(evaluate, tf.float32))
    metrics_accuracy = tf.metrics.accuracy(labels=y, predictions=encoder)

    

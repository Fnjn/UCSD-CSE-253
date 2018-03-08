#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

def IND_z(x, is_training=True):
    # Expected input dims: n * 64 * 64 * 3

    conv1 = tf.contrib.layers.conv2d(x, 32, [5,5], stride=[2,2], padding='VALID')
    bn1 = tf.contrib.layers.batch_norm(conv1, is_training=True)

    conv2 = tf.contrib.layers.conv2d(bn1, 64, [5,5], stride=[2,2], padding='VALID')
    bn2 = tf.contrib.layers.batch_norm(conv2, is_training=True)

    conv3 = tf.contrib.layers.conv2d(bn2, 128, [5,5], stride=[2,2], padding='VALID')
    bn3 = tf.contrib.layers.batch_norm(conv3, is_training=True)

    conv4 = tf.contrib.layers.conv2d(bn3, 256, [5,5], stride=[2,2], padding='VALID')
    bn4 = tf.contrib.layers.batch_norm(conv4, is_training=True)

    bn4_flatten = tf.flatten(bn4)
    fc1 = tf.contrib.layers.fully_connected(bn4_flatten, 4096)
    fc2 = tf.contrib.layers.fully_connected(fc1, 100, activation_fn=None)

    return fc1

def IND_y(x, n_classes, is_training=True):
    # Expected input dims: n * 64 * 64 * 3

    conv1 = tf.contrib.layers.conv2d(x, 32, [5,5], stride=[2,2], padding='VALID')
    bn1 = tf.contrib.layers.batch_norm(conv1, is_training=True)

    conv2 = tf.contrib.layers.conv2d(bn1, 64, [5,5], stride=[2,2], padding='VALID')
    bn2 = tf.contrib.layers.batch_norm(conv2, is_training=True)

    conv3 = tf.contrib.layers.conv2d(bn2, 128, [5,5], stride=[2,2], padding='VALID')
    bn3 = tf.contrib.layers.batch_norm(conv3, is_training=True)

    conv4 = tf.contrib.layers.conv2d(bn3, 256, [5,5], stride=[2,2], padding='VALID')
    bn4 = tf.contrib.layers.batch_norm(conv4, is_training=True)

    bn4_flatten = tf.flatten(bn4)
    fc1 = tf.contrib.layers.fully_connected(bn4_flatten, 512)
    fc2 = tf.contrib.layers.fully_connected(fc1, n_classes, activation_fn=None)

    return fc1

def generator(x, is_training=True):
    # Expected input dims: n * 1 * 1 * (100 + n_classes)

    deconv1 = tf.contrib.layers.conv2d_transpose(x, 512, [4,4], stride=[2,2], padding='VALID')
    bn1 = tf.contrib.layers.batch_norm(deconv1, is_training=True)

    deconv2 = tf.contrib.layers.conv2d_transpose(bn1, 256, [4,4], stride=[2,2], padding='VALID')
    bn2 = tf.contrib.layers.batch_norm(deconv2, is_training=True)

    deconv3 = tf.contrib.layers.conv2d_transpose(bn2, 128, [4,4], stride=[2,2], padding='VALID')
    bn3 = tf.contrib.layers.batch_norm(deconv3, is_training=True)

    deconv4 = tf.contrib.layers.conv2d_transpose(bn3, 64, [4,4], stride=[2,2], padding='VALID')
    bn4 = tf.contrib.layers.batch_norm(deconv4, is_training=True)

    deconv5 = tf.contrib.layers.conv2d_transpose(bn4, 3, [4,4], stride=[2,2], padding='VALID', activation_fn=tf.nn.tanh)

    return deconv5

def discriminator(x, y, is_training=True):
    # Expected input dims: n * 64 * 64 * 3 & n * 32 * 32 * n_classes

    conv1 = tf.contrib.layers.conv2d(x, 64, [4,4], stride=[2,2], padding='VALID', activation_fn=tf.nn.leaky_relu)
    conv1_concat = tf.concat([conv1, y], axis=-1)

    conv2 = tf.contrib.layers.conv2d(conv1_concat, 128, [4,4], stride=[2,2], padding='VALID', activation_fn=tf.nn.leaky_relu)
    bn2 = tf.contrib.layers.batch_norm(conv2, is_training=True)

    conv3 = tf.contrib.layers.conv2d(bn2, 256, [4,4], stride=[2,2], padding='VALID', activation_fn=tf.nn.leaky_relu)
    bn3 = tf.contrib.layers.batch_norm(conv3, is_training=True)

    conv4 = tf.contrib.layers.conv2d(bn3, 512, [4,4], stride=[2,2], padding='VALID', activation_fn=tf.nn.leaky_relu)
    bn4 = tf.contrib.layers.batch_norm(conv4, is_training=True)

    conv5 = tf.contrib.layers.conv2d(bn4, 1, [4,4], stride=[1,1], padding='VALID', activation_fn=tf.nn.sigmoid)

    return conv5

def icgan(x, y):
    # Expected input dims: n * 64 * 64 * 3 & n * n_classes

    with tf.variable_scope('Encoder_Z'):
        Ez = IND_z(x)

    with tf.variable_scope('Encoder_Y'):
        Ey = IND_y(x)

    # TODO: Modify Y

    YZ_concat = tf.concat([Ez, Ey], axis=0) # TODO Check Ez Ey dims

    with tf.variable_scope('Generator'):
        gen = generator(YZ_concat)

# TODO combine discriminator

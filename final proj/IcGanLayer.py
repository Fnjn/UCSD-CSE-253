#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

def IND_z(x, is_training=True):
    # Expected input dims: n * 64 * 64 * 3
    with tf.variable_scope('Encoder_Z', reuse=tf.AUTO_REUSE):
        
        conv1 = tf.contrib.layers.conv2d(x, 32, [5,5], stride=[2,2], padding='VALID')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training=True)

        conv2 = tf.contrib.layers.conv2d(bn1, 64, [5,5], stride=[2,2], padding='VALID')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=True)

        conv3 = tf.contrib.layers.conv2d(bn2, 128, [5,5], stride=[2,2], padding='VALID')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=True)

        conv4 = tf.contrib.layers.conv2d(bn3, 256, [5,5], stride=[2,2], padding='VALID')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=True)

        bn4_flatten = tf.contrib.layers.flatten(bn4)
        fc1 = tf.contrib.layers.fully_connected(bn4_flatten, 4096)
        fc2 = tf.contrib.layers.fully_connected(fc1, 100, activation_fn=None) # dims n * 100

        return fc1

def IND_y(x, ny, is_training=True):
    # Expected input dims: n * 64 * 64 * 3
    with tf.variable_scope('Encoder_Y', reuse=tf.AUTO_REUSE):
        conv1 = tf.contrib.layers.conv2d(x, 32, [5,5], stride=[2,2], padding='VALID')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training=True)

        conv2 = tf.contrib.layers.conv2d(bn1, 64, [5,5], stride=[2,2], padding='VALID')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=True)

        conv3 = tf.contrib.layers.conv2d(bn2, 128, [5,5], stride=[2,2], padding='VALID')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=True)

        conv4 = tf.contrib.layers.conv2d(bn3, 256, [5,5], stride=[2,2], padding='VALID')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=True)

        bn4_flatten = tf.contrib.layers.flatten(bn4)
        fc1 = tf.contrib.layers.fully_connected(bn4_flatten, 512)
        fc2 = tf.contrib.layers.fully_connected(fc1, ny, activation_fn=None) # dims n * ny

    return fc2

def generator(z, y, is_training=True):
    # Expected input dims: n * 100 & n * ny
    with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
        concat = tf.concat([z, y], 1)
        x = tf.expand_dims(tf.expand_dims(concat, 1), 2) # dims: n * 1 * 1 * (100 + ny)
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])

        deconv1 = tf.contrib.layers.conv2d_transpose(x, 512, [4,4], stride=[2,2], padding='VALID')
        bn1 = tf.contrib.layers.batch_norm(deconv1, is_training=True)

        deconv2 = tf.contrib.layers.conv2d_transpose(bn1, 256, [4,4], stride=[2,2], padding='VALID')
        bn2 = tf.contrib.layers.batch_norm(deconv2[:,1:-1,1:-1,:], is_training=True)

        deconv3 = tf.contrib.layers.conv2d_transpose(bn2, 128, [4,4], stride=[2,2], padding='VALID')
        bn3 = tf.contrib.layers.batch_norm(deconv3[:,1:-1,1:-1,:], is_training=True)

        deconv4 = tf.contrib.layers.conv2d_transpose(bn3, 64, [4,4], stride=[2,2], padding='VALID')
        bn4 = tf.contrib.layers.batch_norm(deconv4[:,1:-1,1:-1,:], is_training=True)

        deconv5 = tf.contrib.layers.conv2d_transpose(bn4, 3, [4,4], stride=[2,2], padding='VALID') # dims n * 64 * 64 * 3
        #activation_fn=tf.nn.tanh
        return deconv5[:,1:-1,1:-1,:]

def discriminator(x, y, ny, is_training=True):
    # Expected input dims: n * 64 * 64 * 3 & n * ny

    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
        paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
        conv1 = tf.contrib.layers.conv2d(tf.pad(x, paddings), 64, [4,4], stride=[2,2], padding='VALID', activation_fn=tf.nn.leaky_relu)

        y_duplicated = tf.transpose(tf.reshape(tf.tile(tf.reshape(y, [-1]), tf.constant([32*32])), [32, 32, -1, ny]), perm=[2, 0, 1, 3])
        conv1_concat = tf.concat([conv1, y_duplicated], axis=-1)

        conv2 = tf.contrib.layers.conv2d(tf.pad(conv1_concat, paddings), 128, [4,4], stride=[2,2], padding='VALID', activation_fn=tf.nn.leaky_relu)
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=True)

        conv3 = tf.contrib.layers.conv2d(tf.pad(bn2, paddings), 256, [4,4], stride=[2,2], padding='VALID', activation_fn=tf.nn.leaky_relu)
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=True)

        conv4 = tf.contrib.layers.conv2d(tf.pad(bn3, paddings), 512, [4,4], stride=[2,2], padding='VALID', activation_fn=tf.nn.leaky_relu)
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=True)

        conv5 = tf.contrib.layers.conv2d(bn4, 1, [4,4], stride=[1,1], padding='VALID', activation_fn=None) # dims n * 1 * 1 * 1

        return tf.squeeze(conv5)

def autoencoder_loss(x, y, ny):
    # Expected input dims: n * 64 * 64 * 3 & n * ny
    
    Ez = IND_z(x)
    gen = generator(Ez, y)
        
    loss_autoencoder = tf.reduce_mean(tf.squared_difference(gen, x))
    return loss_autoencoder


def gan_loss(x, y, ny):
    # Expected input dims: n * 64 * 64 * 3 & n * ny

    Ez = IND_z(x)
    gen = generator(Ez, y)


    pred_fake = discriminator(gen, y, ny)
    pred_real = discriminator(x, y, ny)

    loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(pred_fake), logits=pred_fake))
    loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(pred_real), logits=pred_real))

    loss_pred = loss_fake + loss_real
    loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(pred_fake), logits=pred_fake))

    return loss_pred, loss_gen, gen, pred_fake, pred_real

def classify_loss(x, y, ny):
    Ey = IND_y(x, ny, is_training=True)

    loss_cls = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=Ey))
    return loss_cls, Ey

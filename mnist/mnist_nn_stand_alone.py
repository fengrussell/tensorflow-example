# !/usr/bin/env python2
# -*- coding: utf-8 -*-

import tensorflow as tf
# import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

IMAGE_PIXELS = 28
mnist = input_data.read_data_sets("./data/mnist_data/", one_hot=True)
# rng = np.random

X = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS], name='X')
Y = tf.placeholder(tf.float32, [None, 10], name='Y')

W = tf.Variable(tf.zeros([IMAGE_PIXELS * IMAGE_PIXELS, 10]), name='W') # tf.truncated_normal
b = tf.Variable(tf.zeros([10]), name='b')

Y_pred = tf.nn.softmax(tf.matmul(X, W) + b, name='softmax')

cross_entropy = -tf.reduce_sum(Y * tf.log(Y_pred), name='cross_entropy')
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={X: batch_x, Y: batch_y})
    w_value, b_value = sess.run([W, b])
    print(w_value, b_value)
    print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

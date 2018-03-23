# !/usr/bin/env python2
# -*- coding: utf-8 -*-

import tensorflow as tf
# import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/mnist_data/", one_hot=True)
# rng = np.random

IMAGE_PIXELS = 28
batch_size = 128
learning_rate = 0.001
train_step = 100

X = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS], name='X')
Y = tf.placeholder(tf.float32, [None, 10], name='Y')

W = tf.Variable(tf.zeros([IMAGE_PIXELS * IMAGE_PIXELS, 10]), name='W') # tf.truncated_normal
b = tf.Variable(tf.zeros([10]), name='b')

Y_pred = tf.nn.softmax(tf.matmul(X, W) + b, name='softmax')

cross_entropy = -tf.reduce_sum(Y * tf.log(Y_pred), name='cross_entropy')
opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print('num_examples: ' + str(mnist.train.num_examples))

with tf.Session() as sess:
    sess.run(init)
    for i in range(train_step):
        # next_batch每一次读取batch_size长度的数据，如果超过num_examples, 会补齐batch_size, 然后循环读取
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(opt, feed_dict={X: batch_x, Y: batch_y})
    w_value, b_value = sess.run([W, b])
    print(w_value, b_value)
    print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

# print(w_value, b_value)

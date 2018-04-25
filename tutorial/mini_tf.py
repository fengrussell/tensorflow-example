# -*- coding: utf-8 -*-

import tensorflow as tf

from numpy.random import RandomState

batch_size = 8

w = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
b = tf.Variable(tf.random_normal([1], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None, 2), name="x")

y = tf.add(tf.matmul(x, w), b)




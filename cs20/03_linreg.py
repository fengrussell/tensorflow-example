#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 11:52:01 2018

@author: russell
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

DATA_FILE = 'data/slr05.xls'

# step 1. read data from xls
book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)

# cols_x = sheet.col_values(0)
# cols_y = sheet.col_values(1)
# print cols_x, cols_y

data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
#print data
n_samples = sheet.nrows - 1


# step 2. input X and label Y 
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# step 3. create weight and bias
W = tf.Variable(0.0, name='weights')
b = tf.Variable(0.0, name='bias')

# step 4. construct model
Y_pred = X * W + b

# step 5. use the square error as the loss function
loss = tf.square(Y - Y_pred, name='loss')

# step 6. 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    # step 7. initialize variables
    sess.run(tf.global_variables_initializer())
    # step 8. train the model
    for i in range(100):
        for x, y in data:
            sess.run(optimizer, feed_dict={X: x, Y: y})
    # step 9. output the values of W and b
    w_value, b_value = sess.run([W, b])
    print(w_value)
    print(b_value)


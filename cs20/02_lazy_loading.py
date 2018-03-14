#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 11:05:51 2018

@author: russell
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf


### normal loading
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
z = tf.add(x, y)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('graphs/normal_loading', sess.graph)
    
    for _ in range(10):
        sess.run(z)
        
    print(tf.get_default_graph().as_graph_def())
    writer.close()
    
print('--------------------------------------------------------------')
    
### lazy loading
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')    
        
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('graphs/lazy_loading', sess.graph)
    
    for _ in range(10):
        sess.run(tf.add(x, y))
        
    print(tf.get_default_graph().as_graph_def())
    writer.close()
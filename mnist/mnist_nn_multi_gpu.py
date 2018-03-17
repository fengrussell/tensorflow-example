# !/usr/bin/env python2
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/mnist_data/", one_hot=True)

# parameter
IMAGE_PIXELS = 28
batch_size = 128
learning_rate = 0.0005
train_step = 5000
display_step = 100  # 每多少步输出一次结果

n_input = 784
n_classes = 10


# cpu定义的变量
def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape=shape, initializer=initializer, dtype=dtype)
    return var


def build_model(x, y):

    def perceptron(_x, weights, biases):
        return tf.nn.softmax(tf.matmul(_x, weights) + biases, name='softmax')

    with tf.variable_scope('mnist'):
        weight = _variable_on_cpu('W', [n_input, n_classes], tf.random_normal_initializer())
        bias = _variable_on_cpu('b', [n_classes], tf.random_normal_initializer())

        _pred = perceptron(x, weights=weight, biases=bias)
        _coss = -tf.reduce_sum(y * tf.log(_pred), name='cross_entropy')

    return _coss, _pred


def avg_grads(tower_grad):
    average_grads = []
    for grad_and_vars in zip(*tower_grad):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


with tf.Graph().as_default(), tf.device('/cpu:0'):
    X = tf.placeholder(tf.float32, [None, n_input])
    Y = tf.placeholder(tf.float32, [None, n_classes])

    tower_grads = []
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    with tf.variable_scope(tf.get_variable_scope()) as scope:
        for i in xrange(1):
            with tf.device('/cpu:%d' % i): # 没有GPU，先用cpu替代
                cost, pred = build_model(X, Y)
                scope.reuse_variables()
                grads = optimizer.compute_gradients(cost)
                tower_grads.append(grads)

    grads = avg_grads(tower_grads)
    apply_gradient_op = optimizer.apply_gradients(grads)
    train_op = apply_gradient_op

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for step in xrange(train_step):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, cost_print = sess.run([train_op, cost], {X: batch_x, Y: batch_y})

        if step % display_step == 0:
            print("step=%04d" % (step+1) + " cost=" + str(cost_print))

    print("Optimization Finished!")
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    with sess.as_default():
        print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))

    sess.close()

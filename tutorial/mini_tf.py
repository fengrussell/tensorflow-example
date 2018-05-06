# -*- coding: utf-8 -*-
# 定义一个最简单的模型，来分析tf的变量分配已经在分布式下的交互流程

import tensorflow as tf

# batch_size = 8
# W = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1), name="W")
W = tf.Variable(tf.ones([2, 1]), name="W")
b = tf.Variable(tf.zeros([1, 1]), name="b")


def network_fn(input_x):
    # with tf.variable_scope('layer'):
    #
    pred = tf.add(tf.matmul(input_x, W), b)
    return pred


def model_fn(input_x, label):
    pred = network_fn(input_x)
    loss = tf.losses.mean_squared_error(label, pred)
    # train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
    # tf.train.AdamOptimizer(0.01).minimize(loss)
    return loss


def main(_):
    print(tf.get_default_graph())
    x = tf.constant([[1.0, 2.0]])
    y = tf.constant([[1.0]])
    init_op = tf.global_variables_initializer()
    # init_l = tf.local_variables_initializer()
    loss = model_fn(x, y)

    # train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

    with tf.Session() as sess:
        # init_op = tf.global_variables_initializer()
        # train_op =
        # sess.run(tf.initialize_all_variables())
        train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
        sess.run(init_op)

        # sess.run(init_l)

        for i in range(3):
            print(sess.run(W))
            sess.run(train_op)
            # print(sess.run(W))
            # print(sess.run(W))
            # sess.run(train_op)
            # print(sess.run(W))
            # print(sess.run(b))


if __name__ == "__main__":
    tf.app.run()

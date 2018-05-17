# -*- coding: utf-8 -*-
# 定义一个最简单的模型，来分析tf的变量分配已经在分布式下的交互流程

import tensorflow as tf


def _zip_grad_and_vars(grads):
    print(grads)
    print(zip(*grads))


# 最简单的模型, 输入[1 x 2], 矩阵[2 x 1], bias[1], 输出结果[1].
def network_fn(input_x):
    # W = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1), name="W")
    weight = tf.Variable(tf.ones([2, 1]), name="weight")
    bias = tf.Variable(tf.zeros([1]), name="bias")
    return tf.add(tf.matmul(input_x, weight), bias), (weight, bias)


# 模型, 返回train_op. 为了方便查看参数, 会把参数也返回.
# def model_fn(input_x, label):
#     pred, (_weight, _bias) = network_fn(input_x)
#     loss = tf.losses.mean_squared_error(label, pred)
#     train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
#     # train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
#     return train_op, loss, (_weight, _bias)
def model_fn(input_x, label):
    pred, (_weight, _bias) = network_fn(input_x)
    loss = tf.losses.mean_squared_error(label, pred)
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    grads = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads)
    return train_op, loss, grads, (_weight, _bias)


def main(_):
    # 输入参数
    x = tf.constant([[1.0, 2.0]])
    y = tf.constant([[1.0]])

    train_op, loss, grads, (_weight, _bias) = model_fn(x, y)
    # 注意这一行必须要在train_op后面, 否则会有错误Attempting to use uninitialized value weight.
    # 详细可以看这个链接 https://github.com/tensorflow/tensorflow/issues/8057#issuecomment-310505897
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        # print("W[0]:\n %s\nb[0]:%s\n" % (sess.run(_weight), sess.run(_bias)))
        # print("Grads[%d]:%s\n" % (0, sess.run(grads)))

        for i in range(3):
            print("Grads[%d]:%s\n" % (i + 1, sess.run(grads)))
            sess.run(train_op)
            print("W[%d]:\n %s\nb[%d]:%s" % (i+1, sess.run(_weight), i+1, sess.run(_bias)))
            print("loss[%d]:%s" % (i+1, sess.run(loss)))

            # _zip_grad_and_vars(grads)


if __name__ == "__main__":
    tf.app.run()

# -*- coding: utf-8 -*-
# https://github.com/uber/horovod
# https://eng.uber.com/horovod/

import tensorflow as tf
import horovod.tensorflow as hvd
# learn = tf.contrib.learn
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

REGULARAZTION_RATE = 0.0001
DATA_PATH = "../data/mnist"

tf.logging.set_verbosity(tf.logging.INFO)


def _get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def model_fn(image, label):
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

    with tf.variable_scope('layer1'):
        weights = _get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(image, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = _get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    # 计算交叉熵
    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=layer2, labels=tf.argmax(label, 1)))

    return tf.argmax(layer2, 1), cross_entropy


def main(_):
    hvd.init()

    mnist_data = input_data.read_data_sets(DATA_PATH, one_hot=True)
    # mnist = learn.datasets.mnist.read_data_sets('MNIST-data-%d' % hvd.rank())

    with tf.name_scope("input"):
        image = tf.placeholder(tf.float32, [None, INPUT_NODE], name='image')
        label = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='label')

    predict, loss = model_fn(image, label)

    opt = tf.train.RMSPropOptimizer(0.001 * hvd.size())
    opt = hvd.DistributedOptimizer(opt)

    global_step = tf.contrib.framework.get_or_create_global_step()
    train_op = opt.minimize(loss, global_step=global_step)

    hooks = [
            hvd.BroadcastGlobalVariablesHook(0),

            tf.train.StopAtStepHook(last_step=1000 // hvd.size()),
            tf.train.LoggingTensorHook(tensors={'step': global_step, 'loss': loss},
                                       every_n_iter=10),
    ]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    checkpoint_dir = './checkpoints' if hvd.rank() == 0 else None

    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                           hooks=hooks,
                                           config=config) as mon_sess:
        while not mon_sess.should_stop():
            # Run a training step synchronously.
            image_, label_ = mnist_data.train.next_batch(100)
            # image_, label_ = mnist.train.next_batch(100)
            mon_sess.run(train_op, feed_dict={image: image_, label: label_})


if __name__ == "__main__":
    tf.app.run()

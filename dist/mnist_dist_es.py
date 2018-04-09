# -*- coding: utf-8 -*-
# https://gist.github.com/protoget/2cf2b530bc300f209473374cf02ad829 这个代码没有设置Environment.CLOUD，所以worker不会执行
# 目前这个代码只能异步的方式执行，如果是同步的方式参考cifar10_main.py代码，gpu还是要显式的分配op
# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10_estimator/cifar10_main.py

from datetime import datetime
import os
import time
import json

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.estimators import run_config as run_config_lib

# 配置神经网络的参数。
# BATCH_SIZE = 16
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
# TRAINING_STEPS = 10

MODEL_SAVE_PATH = "log/sync"
# DATA_PATH = "../data/mnist"
DATA_PATH = "../data/mnist/train.tfrecords"

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
BATCH_SIZE = 128

tf.logging.set_verbosity(tf.logging.INFO)


def _get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def model_fn(features, labels, mode):
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

    with tf.variable_scope('layer1'):
        weights = _get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(features, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = _get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    # Predicition
    # predictions = {
    #     'classes': tf.argmax(input=layer2, axis=1, name='classes'),
    #     'probabilities': tf.nn.softmax(layer2, name='softmax_tensor')
    # }

    # loss = tf.reduce_mean(
    #     tf.nn.sparse_softmax_cross_entropy_with_logits(logits=layer2, labels=tf.argmax(labels, 1)))
    cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=layer2, labels=labels))
    regulaization_loss = tf.add_n(tf.get_collection('losses'))
    loss = cross_entropy + regulaization_loss

    global_step = tf.train.get_global_step()
    logging_hook = tf.train.LoggingTensorHook({"step": global_step, "loss": loss}, every_n_iter=20)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(0.005)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=global_step)

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])


def train_input_fn():
    dataset = tf.data.TFRecordDataset([DATA_PATH])

    # 定义数据解析格式。
    def parser(record):
        features = tf.parse_single_example(
            record,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'pixels': tf.FixedLenFeature([], tf.int64),
                'label': tf.FixedLenFeature([], tf.int64),
            })

        # 解析图片和标签信息。
        decoded_image = tf.decode_raw(features['image_raw'], tf.uint8)
        reshaped_image = tf.reshape(decoded_image, [784])
        retyped_image = tf.cast(reshaped_image, tf.float32)
        label = tf.cast(features['label'], tf.int32)

        return retyped_image, label

    # 定义输入队列。
    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat(10)
    dataset = dataset.batch(BATCH_SIZE)
    iterator = dataset.make_one_shot_iterator()

    features, labels = iterator.get_next()
    return features, labels


def estimator_fn(config):
    # config = tf.ConfigProto(allow_soft_placement=True)
    # return tf.estimator.Estimator(model_fn=model_fn, model_dir=MODEL_SAVE_PATH,
    #                               config=tf.estimator.RunConfig(session_config=config))
    return tf.estimator.Estimator(model_fn=model_fn, model_dir=MODEL_SAVE_PATH, config=config)


# def experiment_fn(config):
#     return tf.contrib.learn.Experiment(
#         estimator=estimator_fn(config),
#         train_input_fn=train_input_fn,
#         eval_input_fn=train_input_fn,
#         train_steps=1000,
#         eval_steps=100,
#         continuous_eval_throttle_secs=15,
#         eval_delay_secs=10)


def create_experiment_fn(config):
    def experiment_fn(output_dir):
        return tf.contrib.learn.Experiment(
            estimator=estimator_fn(config),
            train_input_fn=train_input_fn,
            eval_input_fn=train_input_fn,
            train_steps=1000,
            eval_steps=100,
            continuous_eval_throttle_secs=15,
            eval_delay_secs=10)
    return experiment_fn


def main(args):
    tf_config = {
        "cluster": {
            'ps': ['127.0.0.1:2222'],
            'worker': ['127.0.0.1:2223', '127.0.0.1:2224']
        }
        # 'environment': 'cloud' # 在tf_config设置enviroment没有效果，还需要通过run_config设置
    }

    if args.type == "worker0":
        tf_config["task"] = {'type': 'worker', 'index': 0}
    elif args.type == "worker1":
        tf_config["task"] = {'type': 'worker', 'index': 1}
    else:
        tf_config["task"] = {'type': 'ps', 'index': 0}

    os.environ['TF_CONFIG'] = json.dumps(tf_config)
    config = run_config_lib.RunConfig()
    config._environment = run_config_lib.Environment.CLOUD  # 需要设置enviroment为cloud才可以按分布式执行

    learn_runner.run(experiment_fn=create_experiment_fn(config), output_dir=MODEL_SAVE_PATH)

    # esti = tf.estimator.Estimator(model_fn=model_fn, model_dir=MODEL_SAVE_PATH,
    #                               config=tf.estimator.RunConfig(session_config=config))

    # Train
    # esti.train(input_fn=train_input_fn, max_steps=1000)

    # Evaluation
    # eval_results = esti.evaluate(input_fn=train_input_fn)
    # print(eval_results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str)

    args = parser.parse_args()
    main(args)


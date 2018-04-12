#! -*- coding: utf-8 -*-
# https://stackoverflow.com/questions/47867748/transfer-learning-with-tf-estimator-estimator-framework

import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import numpy as np

import inception_resnet_v2

NUM_CLASSES = 900
IMAGE_SIZE = 299


def input_fn(mode, num_classes, batch_size=1):
    # some code that loads images, reshapes them to 299x299x3 and batches them
    return tf.constant(np.zeros([batch_size, 299, 299, 3], np.float32)), tf.one_hot(
        tf.constant(np.zeros([batch_size], np.int32)), NUM_CLASSES)


def model_fn(images, labels, num_classes, mode):
    with tf.contrib.slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        logits, end_points = inception_resnet_v2.inception_resnet_v2(images,
                                                                     num_classes,
                                                                     is_training=(mode == tf.estimator.ModeKeys.TRAIN))
    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
    variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
    scopes = {os.path.dirname(v.name) for v in variables_to_restore}

    # tf.train.init_from_checkpoint('inception_resnet_v2_2016_08_30.ckpt',
    #                               {s+'/':s+'/' for s in scopes})
    # 下面是修改后的代码
    tf.train.init_from_checkpoint('inception_resnet_v2_2016_08_30.ckpt',
                                  {v.name.split(':')[0]: v for v in variables_to_restore})

    tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
    total_loss = tf.losses.get_total_loss()  # obtain the regularization losses as well

    # Configure the training op
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(learning_rate=0.00002)
        train_op = optimizer.minimize(total_loss, global_step)
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=total_loss,
        train_op=train_op)


def main(unused_argv):
    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=lambda features, labels, mode: model_fn(features, labels, NUM_CLASSES, mode),
        model_dir='model/MCVE')

    # Train the model
    classifier.train(
        input_fn=lambda: input_fn(tf.estimator.ModeKeys.TRAIN, NUM_CLASSES, batch_size=1),
        steps=1000)

    # Evaluate the model and print results
    eval_results = classifier.evaluate(
        input_fn=lambda: input_fn(tf.estimator.ModeKeys.EVAL, NUM_CLASSES, batch_size=1))
    print()
    print('Evaluation results:\n    %s' % eval_results)


if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])

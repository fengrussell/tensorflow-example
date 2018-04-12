#! -*- coding: utf-8 -*-
# https://github.com/tensorflow/tensorflow/issues/17159 注意有个bug在1.6版本修改的

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import numpy as np
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)


FLAGS = None


def main(unused_args):
    model_dir = os.path.join(FLAGS.model_dir, 'my_model')

    X_train = np.random.rand(30000, 500)
    y_train = np.random.rand(30000, 2048)

    print('Train data shape...')
    print(X_train.shape)
    print(y_train.shape)

    X_dim = X_train.shape[1]
    y_dim = y_train.shape[1]

    feature_columns = [
        tf.feature_column.numeric_column('x', shape=X_dim)
    ]

    warm_start_from = None
    if FLAGS.warm_start:
        warm_start_from = tf.estimator.WarmStartSettings(
            ckpt_to_initialize_from=FLAGS.warm_start,
        )

    model = tf.estimator.DNNRegressor(
        hidden_units=[500, 1000, 2048],
        feature_columns=feature_columns,
        label_dimension=y_dim,
        model_dir=model_dir,
        warm_start_from=warm_start_from
    )

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': X_train},
        y=y_train,
        batch_size=100,
        num_epochs=5,
        shuffle=True
    )

    model.train(input_fn=train_input_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model_dir',
        nargs='?',
        default='output',
        help='Directory to save resulting model(s).'
    )
    parser.add_argument(
        '--warm_start',
        help='Checkpoint to initialize weights from for fine-tuning.'
    )

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
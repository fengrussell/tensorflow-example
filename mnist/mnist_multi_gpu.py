# coding=utf-8
# https://github.com/caicloud/tensorflow-tutorial/blob/master/Deep_Learning_with_TensorFlow/1.4.0/Chapter12/2.%20%E5%A4%9AGPU%E5%B9%B6%E8%A1%8C.py

from datetime import datetime
import os
import time

import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):

        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2


# 定义训练神经网络时需要用到的参数。
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 1000
MOVING_AVERAGE_DECAY = 0.99
N_GPU = 1

# 定义日志和模型输出的路径。
# MODEL_SAVE_PATH = "logs_and_models/"
MODEL_SAVE_PATH = "./log"
MODEL_NAME = "model.ckpt"


# 定义数据存储的路径。因为需要为不同的GPU提供不同的训练数据，所以通过placeholder
# 的方式就需要手动准备多份数据。为了方便训练数据的获取过程，可以采用第7章中介绍的Dataset
# 的方式从TFRecord中读取数据。于是在这里提供的数据文件路径为将MNIST训练数据
# 转化为TFRecords格式之后的路径。如何将MNIST数据转化为TFRecord格式在第7章中有
# 详细介绍，这里不再赘述。
DATA_PATH = "../data/mnist/train.tfrecords"


def get_input():
    dataset = tf.contrib.data.TFRecordDataset([DATA_PATH])

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


def get_loss(x, y_, regulaizer, scope, reuse_variables=None):
    # 计算前向的结果
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        y = inference(x, regulaizer)
    # 计算交叉熵
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=y_))

    regulaization_loss = tf.add_n(tf.get_collection('losses', scope))

    loss = cross_entropy + regulaization_loss
    return loss


def average_gradients(tower_grads):
    avg_grads = []

    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)

        avg_grads.append(grad_and_var)

    return avg_grads


def main(argv=None):

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        x, y_ = get_input()
        regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 60000 / BATCH_SIZE, LEARNING_RATE_DECAY)

        opt = tf.train.GradientDescentOptimizer(learning_rate)

        tower_grads = []
        reuse_variable = False

        for i in range(N_GPU):
            # with tf.device('/gpu:%d' % i) as scope:
            with tf.device('/cpu:0') as scope:
                cur_loss = get_loss(x, y_, regularizer, scope, reuse_variable)

                reuse_variable = True
                grads = opt.compute_gradients(cur_loss)
                tower_grads.append(grads)

        grads = average_gradients(tower_grads)

        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram('gradients_on_average/%s' % var.op.name, grad)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variables_to_average)

        train_op = tf.group(apply_gradient_op, variables_averages_op)
        # train_op = apply_gradient_op

        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
            init.run()
            summary_writer = tf.summary.FileWriter(MODEL_SAVE_PATH, sess.graph)

            for step in range(TRAINING_STEPS):
                start_time = time.time()
                _, loss_value = sess.run([train_op, cur_loss])

                duration = time.time() - start_time

                if step != 0 and step % 10 == 0:
                    num_examples_per_step = BATCH_SIZE * N_GPU
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / N_GPU

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                    print (format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

                    # 通过TensorBoard可视化训练过程。
                    summary = sess.run(summary_op)
                    summary_writer.add_summary(summary, step)

                if step != 0 and (step % 1000 == 0 or (step + 1) == TRAINING_STEPS):
                    checkpoint_path = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)
                    saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':
    tf.app.run()


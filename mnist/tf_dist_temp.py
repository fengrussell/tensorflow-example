# -*- coding: utf-8 -*-

from datetime import datetime
import os
import time

import tensorflow as tf


# 设定参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 20000
MOVING_AVERAGE_DECAY = 0.99

# 模型保存路径
MODEL_SAVE_PATH = "./log/sync"
# 数据路径
DATA_PATH = "../data/mnist/train.tfrecords"

# gpu数量，先定义为常量
#N_GPU = 2
N_GPU = 1


# 运行程序传入的参数
FLAGS = tf.app.flags.FLAGS

# ps or worker
tf.app.flags.DEFINE_string('job_name', 'worker', ' "ps" or "worker" ')
# 集群中参数服务器的地址
tf.app.flags.DEFINE_string(
    'ps_hosts', ' tf-ps0:2222,tf-ps1:1111',
    'Comma-separated list of hostname:port for the parameter server jobs. e.g. "tf-ps0:2222,tf-ps1:1111" ')
# 集群中worker服务器的地址
tf.app.flags.DEFINE_string(
    'worker_hosts', ' tf-worker0:2222,tf-worker1:1111',
    'Comma-separated list of hostname:port for the worker jobs. e.g. "tf-worker0:2222,tf-worker1:1111" ')
# 当然程序的任务ID
tf.app.flags.DEFINE_integer('task_id', 0, 'Task ID of the worker/replica running the training.')


INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def read_data():
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


def build_model(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2


def get_loss(x, y, regularizer, scope, reuse_variables=None):
    # 计算前向的结果
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        y_pred = build_model(x, regularizer)

    # 计算交叉熵
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=y))
    # loss = cross_entropy + tf.add_n(tf.get_collection("losses", scope))

    regulaization_loss = tf.add_n(tf.get_collection('losses', scope))

    loss = cross_entropy + regulaization_loss
    return loss


# 定义Graph，同步的方式
# def build_graph(x, y, num_workers, is_chief):
#     #
#     regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
#     y_pred = build_model(x, regularizer)
#
#     # 不是很清楚这种方式和别的代码有什么区别？
#     global_step = tf.contrib.framework.get_or_create_global_step()
#
#     # 计算损失函数并定义反向传播过程。
#     cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=tf.argmax(y, 1))
#     cross_entropy_mean = tf.reduce_mean(cross_entropy)
#     loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
#     learning_rate = tf.train.exponential_decay(
#         LEARNING_RATE_BASE,
#         global_step,
#         60000 / BATCH_SIZE,
#         LEARNING_RATE_DECAY)
#
#
#     # 通过tf.train.SyncReplicasOptimizer函数实现同步更新。
#     opt = tf.train.SyncReplicasOptimizer(
#         tf.train.GradientDescentOptimizer(learning_rate),
#         replicas_to_aggregate=num_workers,
#         total_num_replicas=num_workers)
#     sync_replicas_hook = opt.make_session_run_hook(is_chief)
#     train_op = opt.minimize(loss, global_step=global_step)
#
#     #
#     if is_chief:
#         variable_averages = tf.train.ExponentialMovingAverage(
#             MOVING_AVERAGE_DECAY, global_step)
#         variables_averages_op = variable_averages.apply(
#             tf.trainable_variables())
#         with tf.control_dependencies([variables_averages_op, train_op]):
#             train_op = tf.no_op()
#
#     return global_step, loss, train_op, sync_replicas_hook


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


# 定义task下的graph
def build_task_graph(num_workers, is_chief):
    x, y = read_data()
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

    global_step = tf.contrib.framework.get_or_create_global_step()
    # global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 60000 / BATCH_SIZE, LEARNING_RATE_DECAY)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    tower_grads = []
    loss_list = []
    reuse_variable = False

    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(N_GPU):
            with tf.device('/cpu:%d' % i):
                with tf.name_scope('GPU_%d' % i) as scope:
                    loss = get_loss(x, y, regularizer, scope, reuse_variable)
                    reuse_variable = True

                    loss_list.append(loss)
                    grads = optimizer.compute_gradients(loss)
                    tower_grads.append(grads)

    with tf.device('/cpu:0'):
        # 通过tf.train.SyncReplicasOptimizer函数实现同步更新。
        rep_op = tf.train.SyncReplicasOptimizer(
            optimizer,
            replicas_to_aggregate=num_workers,
            total_num_replicas=num_workers)
        sync_replicas_hook = rep_op.make_session_run_hook(is_chief)

        grads = average_gradients(tower_grads)
        train_op = rep_op.apply_gradients(grads, global_step=global_step)

    if is_chief:
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())
        with tf.control_dependencies([variables_averages_op, train_op]):
            train_op = tf.no_op()

    return global_step, loss_list, train_op, sync_replicas_hook


def main(argv=None):
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    num_workers = len(worker_hosts)
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_id)

    if FLAGS.job_name == 'ps':
        with tf.device("/cpu:0"):
            server.join()

    is_chief = (FLAGS.task_id == 0)
    # mnist = input_data.read_data_sets(DATA_PATH, one_hot=True)

    device_setter = tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_id,
        cluster=cluster)

    with tf.device(device_setter):
        global_step, loss_list, train_op, sync_replicas_hook = build_task_graph(num_workers, is_chief)

        # 把处理同步更新的hook也加进来。
        hooks=[sync_replicas_hook, tf.train.StopAtStepHook(last_step=TRAINING_STEPS)]
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     log_device_placement=False)

        # 训练过程和异步一致。
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=is_chief,
                                               checkpoint_dir=MODEL_SAVE_PATH,
                                               hooks=hooks,
                                               save_checkpoint_secs=60,
                                               config=sess_config) as mon_sess:
            print "session started."
            step = 0
            start_time = time.time()

            while not mon_sess.should_stop():                
                # xs, ys = mnist.train.next_batch(BATCH_SIZE)
                _, loss_value, global_step_value = mon_sess.run(
                    [train_op, loss_list, global_step])

                if step > 0 and step % 100 == 0:
                    duration = time.time() - start_time
                    sec_per_batch = duration / global_step_value
                    format_str = "After %d training steps (%d global steps), " +\
                                 "avg_loss on training batch is [%s]. (%.3f sec/batch)"
                    print format_str % (step, global_step_value, loss_value_str(loss_value), sec_per_batch)
                step += 1


def loss_value_str(loss_value):
    l = ''
    return l.join(map(lambda x: '%f ' % x, loss_value))


if __name__ == "__main__":
    tf.app.run()

# -*- coding: utf-8 -*-
# 执行命令：
# python mnist_dist_spec_gpu.py --ps_hosts=1.1.1.1:2222 --worker_hosts=1.1.1.1:2222,1.1.1.2:2222 --job_name=ps/worker --task_id=0

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

# 配置神经网络的参数。
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 1000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "log/sync"
DATA_PATH = "../data/mnist"

# 和异步模式类似的设置flags。
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('job_name', 'worker', ' "ps" or "worker" ')
tf.app.flags.DEFINE_string(
    'ps_hosts', ' tf-ps0:2222,tf-ps1:1111',
    'Comma-separated list of hostname:port for the parameter server jobs. e.g. "tf-ps0:2222,tf-ps1:1111" ')
tf.app.flags.DEFINE_string(
    'worker_hosts', ' tf-worker0:2222,tf-worker1:1111',
    'Comma-separated list of hostname:port for the worker jobs. e.g. "tf-worker0:2222,tf-worker1:1111" ')
tf.app.flags.DEFINE_integer('task_id', 0, 'Task ID of the worker/replica running the training.')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'How many GPUs to use.')


####################
# Define the model #
####################

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


def _get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


def mnist_model(input_tensor, regularizer):
    with tf.variable_scope('layer1'):

        weights = _get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = _get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2


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
    # 解析参数，定义cluster
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    num_workers = len(worker_hosts)
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_id)

    # 参数服务
    if FLAGS.job_name == 'ps':
        with tf.device("/cpu:0"):
            server.join()  # 阻塞等待worker的请求

    is_chief = (FLAGS.task_id == 0)
    mnist_data = input_data.read_data_sets(DATA_PATH, one_hot=True)

    # 向ps注册graph
    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d") % FLAGS.task_id,
                   cluster=cluster):
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='input_x')
        y = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='input_y')

        regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 60000 / BATCH_SIZE,
                                                   LEARNING_RATE_DECAY)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        tower_grads = []

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(FLAGS.num_gpus):
                with tf.device("/gpu:%d" % i):
                    with tf.name_scope('worker%d_cpu%d' % (FLAGS.task_id, i)) as scope:
                        y_pred = mnist_model(x, regularizer)

                        # 计算交叉熵
                        cross_entropy = tf.reduce_mean(
                            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=y))
                        regulaization_loss = tf.add_n(tf.get_collection('losses', scope))
                        loss = cross_entropy + regulaization_loss

                        # 下一个gpu可以复用这些变量
                        tf.get_variable_scope().reuse_variables()
                        grads = optimizer.compute_gradients(loss)
                        tower_grads.append(grads)


        # 剩下的默认应该是cpu0，待确认
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

        # 把处理同步更新的hook也加进来。
        hooks = [sync_replicas_hook, tf.train.StopAtStepHook(last_step=TRAINING_STEPS)]
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
                xs, ys = mnist_data.train.next_batch(BATCH_SIZE)
                _, loss_value, global_step_value = mon_sess.run(
                    [train_op, loss, global_step], feed_dict={x: xs, y: ys})

                if step > 0 and step % 100 == 0:
                    duration = time.time() - start_time
                    sec_per_batch = duration / global_step_value
                    format_str = "After %d training steps (%d global steps), " + \
                                 "loss on training batch is %g. (%.3f sec/batch)"
                    print format_str % (step, global_step_value, loss_value, sec_per_batch)
                step += 1


if __name__ == "__main__":
    tf.app.run()

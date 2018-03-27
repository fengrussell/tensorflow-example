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


# mnist所用的模型
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


# 构建模型
def build_model(x, y, num_workers, is_chief):
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    # global_step = tf.train.get_or_create_global_step(
    global_step=tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 60000 / BATCH_SIZE, LEARNING_RATE_DECAY)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    # 每个gpu计算的loss、grad的集合
    tower_grads = []
    tower_losses = []

    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(FLAGS.num_gpus):
            with tf.device("/gpu:%d" % i):
                with tf.name_scope('worker%d_gpu%d' % (FLAGS.task_id, i)) as scope:
                    y_pred = mnist_model(x, regularizer)

                    # 计算交叉熵
                    cross_entropy = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=tf.argmax(y, 1)))
                    regulaization_loss = tf.add_n(tf.get_collection('losses', scope))

                    opt_gpu = tf.train.GradientDescentOptimizer(learning_rate)
                    loss = cross_entropy + regulaization_loss
                    # grads = optimizer.compute_gradients(loss)
                    grads = opt_gpu.compute_gradients(loss)

                    tower_losses.append(loss)
                    tower_grads.append(grads)

                    # 下一个gpu可以复用这些变量
                    tf.get_variable_scope().reuse_variables()

    # 剩下的默认应该是cpu0，待确认
    # with tf.device('/cpu:0'):
    rep_op = tf.train.SyncReplicasOptimizer(
        optimizer,
        replicas_to_aggregate=num_workers,
        total_num_replicas=num_workers)
    # sync_replicas_hook = rep_op.make_session_run_hook(is_chief)

    # 计算当前worker的平均梯度
    grads = average_gradients(tower_grads)
    # 用这个梯度值进行apply_gradients
    train_op = rep_op.apply_gradients(grads, global_step=global_step)

    if is_chief:
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())
        with tf.control_dependencies([variables_averages_op, train_op]):
            train_op = tf.no_op()

    return global_step, tower_losses, train_op, rep_op


def main(argv=None):
    # 解析参数，定义cluster
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    num_workers = len(worker_hosts)
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_id)

    # 参数服务器
    if FLAGS.job_name == 'ps':
        with tf.device("/cpu:0"):
            server.join()  # 阻塞等待worker的请求

    is_chief = (FLAGS.task_id == 0)
    mnist_data = input_data.read_data_sets(DATA_PATH, one_hot=True)

    # 向ps注册graph
    # merge_devices如果为True， 由task_id=0的worker统一负责ps的一些操作；如果为False，每个worker都会参与ps的操作，是否冲突或覆盖不是很清楚。
    # log_device_placement=True，打印每个worker上每个操作在哪个设备上，对比一下内容，会发现merge_devices=False，每个worker（between-Graph）是一样的。
    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_id,
                   cluster=cluster)):
        x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='input_x')
        y = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='input_y')
        # 数据的获取在build_model，这个根据实际的情况来决定输入数据在哪处定义
        global_step, losses, train_op, rep_op = build_model(x, y, num_workers, is_chief)

        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()
        init_op = tf.initialize_all_variables()

        # 在同步模式下，主计算服务器需要协调不同计算服务器计算得到的参数梯度并最终更新
        # 参数。这需要主计算服务器完成一些额外的初始化工作。
        if is_chief:
            # 定义协调不同计算服务器的队列并定义初始化操作。
            chief_queue_runner = rep_op.get_chief_queue_runner()
            init_tokens_op = rep_op.get_init_tokens_op(0)

        # 和异步模式类似的声明tf.train.Supervisor。
        sv = tf.train.Supervisor(is_chief=is_chief,
                                 logdir=MODEL_SAVE_PATH,
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=60,
                                 save_summaries_secs=60)
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = sv.prepare_or_wait_for_session(
            server.target, config=sess_config)

        # 在开始训练模型之前，主计算服务器需要启动协调同步更新的队列并执行初始化操作。
        if is_chief:
            sv.start_queue_runners(sess, [chief_queue_runner])
            sess.run(init_tokens_op)

        # 和异步模式类似的运行迭代的训练过程。
        step = 0
        start_time = time.time()
        while not sv.should_stop():
            xs, ys = mnist_data.train.next_batch(BATCH_SIZE)
            _, loss_value, global_step_value = sess.run(
                  [train_op, losses, global_step], feed_dict={x: xs, y: ys})
            if global_step_value >= TRAINING_STEPS: break

            if step > 0 and step % 100 == 0:
                duration = time.time() - start_time
                sec_per_batch = duration / (global_step_value * num_workers)
                format_str = ("After %d training steps (%d global steps), "
                                   "loss on training batch is %g.  " 
                                   "(%.3f sec/batch)")
                print(format_str % (step, global_step_value,
                                    fromat_losses(loss_value), sec_per_batch))
            step += 1
    sv.stop()


def fromat_losses(losses):
    return sum(losses)/len(losses)


if __name__ == "__main__":
    tf.app.run()

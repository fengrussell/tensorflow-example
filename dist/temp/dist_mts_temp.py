# -*- coding: utf-8 -*-
# 利用MonitoredTrainingSession实现tf分布式同步训练

import tensorflow as tf

import time
from datetime import datetime

# 1. 定义shell执行脚本要接收的参数

FLAGS = tf.app.flags.FLAGS

# Cluster Flags
tf.app.flags.DEFINE_string('job_name', 'worker', ' "ps" or "worker" ')
tf.app.flags.DEFINE_string(
    'ps_hosts', ' tf-ps0:2222,tf-ps1:1111',
    'Comma-separated list of hostname:port for the parameter server jobs. e.g. "tf-ps0:2222,tf-ps1:1111" ')
tf.app.flags.DEFINE_string(
    'worker_hosts', ' tf-worker0:2222,tf-worker1:1111',
    'Comma-separated list of hostname:port for the worker jobs. e.g. "tf-worker0:2222,tf-worker1:1111" ')
tf.app.flags.DEFINE_integer('task_id', 0, 'Task ID of the worker/replica running the training.')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'How many GPUs to use.')

# Ckpt Flags
tf.app.flags.DEFINE_string('checkpoint', './checkpoint/', 'Directory where checkpoints and event logs are written to.')
# Fine-Tuning Flags
tf.app.flags.DEFINE_string('pre_trained_model', './pre_trained_model/',
                           'Directory where checkpoints and event logs are written to.')

# Dataset Flags
tf.app.flags.DEFINE_string('dataset_dir', './dataset/', 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer('batch_size', 16, 'The number of samples in each train batch.')
tf.app.flags.DEFINE_integer('max_number_of_steps', 100, 'The maximum number of training steps.')

# Log Flags
tf.app.flags.DEFINE_integer('log_every_n_steps', 5, 'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer('save_summaries_secs', None, 'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer('save_interval_secs', 3600, 'The frequency with which the model is saved, in seconds.')

# Optimization Flags
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
tf.app.flags.DEFINE_float('end_learning_rate', 0.0001,
                          'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float('moving_average_decay', 0.99,
                          'The decay to use for the moving average.',
                          'If left as None, then moving averages are not used.')


# 2. 私有方法

def _sum_gpu_gradients(clone_grads):
    sum_grads = []
    for grad_and_vars in zip(*clone_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad_var0_clone0, var0), ... (grad_varN_cloneN, varN))
        grads = []
        var = grad_and_vars[0][1]
        for g, v in grad_and_vars:
            assert v == var
            if g is not None:
                grads.append(g)
        if grads:
            if len(grads) > 1:
                sum_grad = tf.add_n(grads, name=var.op.name + '/sum_grads')
                sum_grad = tf.div(sum_grad, 1.0 * len(grads))
                # print('mean grad')
            else:
                sum_grad = grads[0]
            sum_grads.append((sum_grad, var))
    return sum_grads


def _sum_losses(losses):
    if len(losses) <= 1:
        return "%g" % losses[0]

    # "avg_loss [g0_loss, g1_loss]"
    return "%g [%s]" % (sum(losses)/len(losses), ", ".join(map(lambda x: '%g' % x, losses)))


def _get_init_fn_of_scaffold(saver, model_path):
    def init_fn(scaffold, session):
        saver.restore(session, model_path)

    return init_fn


# 3. 输入数据queue/函数
def input_queue():
    return 1


# 定义读取数据(tfrecords)的函数，返回一个queue，这种方式方便每个gpu分配数据。（placeholder的方式还需要split）
def input_fn():
    file_names = tf.train.match_filenames_once(FLAGS.dataset_dir)
    filename_queue = tf.train.string_input_producer(file_names)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features['image'], tf.uint8)
    # image = tf.reshape(image, [FLAGS.orgin_height, FLAGS.orgin_width, 3])

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([784])  # 28x28
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    images, labels = tf.train.shuffle_batch([image, label],
                                            batch_size=FLAGS.batch_size,
                                            capacity=FLAGS.batch_size*1000,
                                            min_after_dequeue=FLAGS.batch_size*500,
                                            num_threads=2)

    # 先返回 images, labels
    # queue需要第三方的支持，例如slim，slim.prefetch_queue.prefetch_queue([images, labels], capacity=16, num_threads=2)
    return images, labels


# 4. 网络函数
def network_fn(images):
    return 1


# 5. 模型（同步分布式），计算grad、loss
def model_fn(num_workers, is_chief):

    all_grads = []
    all_losses = []
    global_step = tf.train.get_or_create_global_step()

    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)

    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(FLAGS.num_gpus):
            with tf.device("/gpu:%d" % i):
                with tf.name_scope('worker%d_gpu%d' % (FLAGS.task_id, i)) as scope:
                    images, labels = input_fn()

                    # 计算logits
                    logits = network_fn(images)
                    #
                    cross_entroy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
                    # 计算loss和grad
                    loss = tf.reduce_mean(tf.reduce_mean(cross_entroy))
                    grads = optimizer.compute_gradients(loss)

                    all_losses.append(loss)
                    all_grads.append(grads)

                    # 下一个gpu可以复用这些变量
                    tf.get_variable_scope().reuse_variables()

    with tf.device("/cpu:0"):
        sync_opt = tf.train.SyncReplicasOptimizer(
            optimizer,
            replicas_to_aggregate=num_workers,
            total_num_replicas=num_workers)

        grads = _sum_gpu_gradients(all_grads)
        train_op = sync_opt.apply_gradients(grads, global_step=global_step)

    # 定义hook，目前num_tokens=0才能保证同步执行，不过执行完会抛出一个异常，但不影响Training。
    sync_replicas_hook = sync_opt.make_session_run_hook(is_chief, num_tokens=0)

    # 滑动平均
    if FLAGS.moving_average_decay is not None and is_chief:
        variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        #
        with tf.control_dependencies([variables_averages_op, train_op]):
            train_op = tf.no_op()

    return global_step, all_losses, train_op, sync_replicas_hook


def main(_):
    # log level
    tf.logging.set_verbosity(tf.logging.INFO)

    # 解析参数，定义cluster
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    num_workers = len(worker_hosts)  # worker的数量
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_id)

    # 参数服务器
    if FLAGS.job_name == 'ps':
        with tf.device("/cpu:0"):
            server.join()  # 阻塞等待worker的请求

    # task_id为0代表chief
    is_chief = (FLAGS.task_id == 0)

    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_id,
                                                  cluster=cluster)):

        # 1. 训练数据，显式的指定cpu处理数据，避免一些默认环境变量下在gpu执行
        with tf.device("/cpu:0"):
            batch_fn = input_queue()  # 如果采用queue来传递数据，用这种方式

        # 2. 模型返回op、hook、global_step等
        global_step, losses, train_op, sync_replicas_hook = model_fn(num_workers, is_chief)

        # 3. 定义hooks，保持同步、训练终止
        # StopAtStepHook有两个参数：num_steps、last_step，二选一
        hooks = [sync_replicas_hook, tf.train.StopAtStepHook(last_step=FLAGS.max_number_of_steps)]
        if is_chief:
            # chief定义保存ckpt的hook
            saver_hook = tf.train.CheckpointSaverHook(
                checkpoint_dir=FLAGS.train_dir,
                save_steps=FLAGS.steps_each_save  # save_steps/save_secs参数只能二选一
            )
            hooks.append(saver_hook)

        # 4. scaffold，定义需要做的初始化的op
        scaffold = tf.train.Scaffold(init_op=tf.global_variables_initializer(),
                                     init_fn=_get_init_fn_of_scaffold() if is_chief else None)

        # 5. session
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     log_device_placement=False)

        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=is_chief,
                                               checkpoint_dir=FLAGS.train_dir,
                                               hooks=hooks,
                                               scaffold=scaffold,
                                               save_checkpoint_secs=3600,
                                               config=sess_config) as mon_sess:
            print("session started. ")
            step = 0
            start_time = time.time()

            while not mon_sess.should_stop():
                _, loss_value, global_step_value = mon_sess.run([train_op, losses, global_step])

                if global_step_value >= FLAGS.max_number_of_steps:
                    break

                if step > 0 and step % FLAGS.log_every_n_steps == 0:
                    duration = time.time() - start_time
                    sec_per_step = duration / global_step_value

                    format_str = "%s: global step %d (local step %d), loss = %s (%.3f sec/step) "
                    print(format_str % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), global_step_value,
                                        step, _sum_losses(loss_value), sec_per_step))
                step += 1

            if is_chief:
                print('Finished training! ')


if __name__ == '__main__':
    tf.app.run()

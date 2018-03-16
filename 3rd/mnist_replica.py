# -*- coding: utf-8 -*-
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dist_test/python/mnist_replica.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import tempfile
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 定义常量，用于创建数据流图
flags = tf.app.flags
flags.DEFINE_string("data_dir", "/tmp/mnist-data",
                    "Directory for storing mnist data")
# 只下载数据，不做其他操作
flags.DEFINE_boolean("download_only", False,
                     "Only perform downloading of data; Do not proceed to "
                     "session preparation, mnist definition or training")
# task_index从0开始。0代表用来初始化变量的第一个任务
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
# 每台机器GPU个数，机器没有GPU为0
flags.DEFINE_integer("num_gpus", 1,
                     "Total number of gpus for each machine."
                     "If you don't use GPU, please set it to '0'")
# 同步训练模型下，设置收集工作节点数量。默认工作节点总数
flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before parameter update"
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")
flags.DEFINE_integer("hidden_units", 100,
                     "Number of units in the hidden layer of the NN")
# 训练次数
flags.DEFINE_integer("train_steps", 200,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 100, "Training batch size")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
# 使用同步训练、异步训练
flags.DEFINE_boolean("sync_replicas", False,
                     "Use the sync_replicas (synchronized replicas) mode, "
                     "wherein the parameter updates from workers are aggregated "
                     "before applied to avoid stale gradients")
# 如果服务器已经存在，采用gRPC协议通信；如果不存在，采用进程间通信
flags.DEFINE_boolean(
    "existing_servers", False, "Whether servers already exists. If True, "
    "will use the worker hosts via their GRPC URLs (one client process "
    "per worker host). Otherwise, will create an in-process TensorFlow "
    "server.")
# 参数服务器主机
flags.DEFINE_string("ps_hosts","localhost:2222",
                    "Comma-separated list of hostname:port pairs")
# 工作节点主机
flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224",
                    "Comma-separated list of hostname:port pairs")
# 本作业是工作节点还是参数服务器
flags.DEFINE_string("job_name", None,"job name: worker or ps")

FLAGS = flags.FLAGS
IMAGE_PIXELS = 28


def main(unused_argv):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    if FLAGS.download_only:
        sys.exit(0)
    if FLAGS.job_name is None or FLAGS.job_name == "":
        raise ValueError("Must specify an explicit `job_name`")
    if FLAGS.task_index is None or FLAGS.task_index == "":
        raise ValueError("Must specify an explicit `task_index`")

    print("job name = %s" % FLAGS.job_name)
    print("task index = %d" % FLAGS.task_index)

    # Construct the cluster and start the server
    # 读取集群描述信息
    ps_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")

    # Get the number of workers.
    num_workers = len(worker_spec)

    # 创建TensorFlow集群描述对象
    cluster = tf.train.ClusterSpec({
        "ps": ps_spec,
        "worker": worker_spec})

    # 为本地执行任务创建TensorFlow Server对象。
    if not FLAGS.existing_servers:
        # Not using existing servers. Create an in-process server.
        # 创建本地Sever对象，从tf.train.Server这个定义开始，每个节点开始不同
        # 根据执行的命令的参数(作业名字)不同，决定这个任务是哪个任务
        # 如果作业名字是ps，进程就加入这里，作为参数更新的服务，等待其他工作节点给它提交参数更新的数据
        # 如果作业名字是worker，就执行后面的计算任务
        server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
        # 如果是参数服务器，直接启动即可。这里，进程就会阻塞在这里
        # 下面的tf.train.replica_device_setter代码会将参数传给ps_server保管
        if FLAGS.job_name == "ps":
            server.join()
    # 处理工作节点
    # 找出worker的主节点，即task_index为0的点
    is_chief = (FLAGS.task_index == 0)

    # 如果使用gpu
    if FLAGS.num_gpus > 0:
        # Avoid gpu allocation conflict: now allocate task_num -> #gpu
        # for each worker in the corresponding machine
        gpu = (FLAGS.task_index % FLAGS.num_gpus)
        # 分配worker到指定gpu上运行
        worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, gpu)
    # 如果使用cpu
    elif FLAGS.num_gpus == 0:
        # Just allocate the CPU to worker server
        # 把cpu分配给worker
        cpu = 0
        worker_device = "/job:worker/task:%d/cpu:%d" % (FLAGS.task_index, cpu)

    # The device setter will automatically place Variables ops on separate
    # parameter servers (ps). The non-Variable ops will be placed on the workers.
    # The ps use CPU and workers use corresponding GPU
    # 用tf.train.replica_device_setter将涉及变量操作分配到参数服务器上，使用CPU。将涉及非变量操作分配到工作节点上，使用上一步worker_device值。
    # 在这个with语句之下定义的参数，会自动分配到参数服务器上去定义。如果有多个参数服务器，就轮流循环分配
    with tf.device(
            tf.train.replica_device_setter(
                worker_device=worker_device,
                ps_device="/job:ps/cpu:0",
                cluster=cluster)):

        # 定义全局步长，默认值为0
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # Variables of the hidden layer
        # 定义隐藏层参数变量，这里是全连接神经网络隐藏层
        hid_w = tf.Variable(
            tf.truncated_normal(
                [IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
                stddev=1.0 / IMAGE_PIXELS),
            name="hid_w")
        hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")

        # Variables of the softmax layer
        # 定义Softmax 回归层参数变量
        sm_w = tf.Variable(
            tf.truncated_normal(
                [FLAGS.hidden_units, 10],
                stddev=1.0 / math.sqrt(FLAGS.hidden_units)),
            name="sm_w")
        sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

        # Ops: located on the worker specified with FLAGS.task_index
        # 定义模型输入数据变量
        x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
        y_ = tf.placeholder(tf.float32, [None, 10])

        # 构建隐藏层
        hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
        hid = tf.nn.relu(hid_lin)

        # 构建损失函数和优化器
        y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
        cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

        # 异步训练模式：自己计算完成梯度就去更新参数，不同副本之间不会去协调进度
        opt = tf.train.AdamOptimizer(FLAGS.learning_rate)

        # 同步训练模式
        if FLAGS.sync_replicas:
            if FLAGS.replicas_to_aggregate is None:
                replicas_to_aggregate = num_workers
            else:
                replicas_to_aggregate = FLAGS.replicas_to_aggregate
            # 使用SyncReplicasOptimizer作优化器，并且是在图间复制情况下
            # 在图内复制情况下将所有梯度平均
            opt = tf.train.SyncReplicasOptimizer(
                opt,
                replicas_to_aggregate=replicas_to_aggregate,
                total_num_replicas=num_workers,
                name="mnist_sync_replicas")
        train_step = opt.minimize(cross_entropy, global_step=global_step)

        if FLAGS.sync_replicas:
            local_init_op = opt.local_step_init_op
            if is_chief:
                # 所有进行计算工作节点里一个主工作节点(chief)
                # 主节点负责初始化参数、模型保存、概要保存
                local_init_op = opt.chief_init_op
            ready_for_local_init_op = opt.ready_for_local_init_op
            # Initial token and chief queue runners required by the sync_replicas mode
            # 同步训练模式所需初始令牌、主队列
            chief_queue_runner = opt.get_chief_queue_runner()
            sync_init_op = opt.get_init_tokens_op()
        init_op = tf.global_variables_initializer()
        train_dir = tempfile.mkdtemp()

        if FLAGS.sync_replicas:
            # 创建一个监管程序，用于统计训练模型过程中的信息
            # lodger 是保存和加载模型路径
            # 启动就会去这个logdir目录看是否有检查点文件，有的话就自动加载
            # 没有就用init_op指定初始化参数
            # 主工作节点(chief)负责模型参数初始化工作
            # 过程中，其他工作节点等待主节点完成初始化工作，初始化完成后，一起开始训练数据
            # global_step值是所有计算节点共享的
            # 在执行损失函数最小值时自动加1，通过global_step知道所有计算节点一共计算多少步
            sv = tf.train.Supervisor(
                is_chief=is_chief,
                logdir=train_dir,
                init_op=init_op,
                local_init_op=local_init_op,
                ready_for_local_init_op=ready_for_local_init_op,
                recovery_wait_secs=1,
                global_step=global_step)
        else:
            sv = tf.train.Supervisor(
                is_chief=is_chief,
                logdir=train_dir,
                init_op=init_op,
                recovery_wait_secs=1,
                global_step=global_step)

        # 创建会话，设置属性allow_soft_placement为True
        # 所有操作默认使用被指定设置，如GPU
        # 如果该操作函数没有GPU实现，自动使用CPU设备
        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index])

        # The chief worker (task_index==0) session will prepare the session,
        # while the remaining workers will wait for the preparation to complete.
        # 主工作节点(chief)，task_index为0节点初始化会话
        # 其余工作节点等待会话被初始化后进行计算
        if is_chief:
            print("Worker %d: Initializing session..." % FLAGS.task_index)
        else:
            print("Worker %d: Waiting for session to be initialized..." %
                  FLAGS.task_index)

        if FLAGS.existing_servers:
            server_grpc_url = "grpc://" + worker_spec[FLAGS.task_index]
            print("Using existing server at: %s" % server_grpc_url)
            # 创建TensorFlow会话对象，用于执行TensorFlow图计算
            # prepare_or_wait_for_session需要参数初始化完成且主节点准备好后，才开始训练
            sess = sv.prepare_or_wait_for_session(server_grpc_url, config=sess_config)
        else:
            sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)
        print("Worker %d: Session initialization complete." % FLAGS.task_index)

        if FLAGS.sync_replicas and is_chief:
            # Chief worker will start the chief queue runner and call the init op.
            sess.run(sync_init_op)
            sv.start_queue_runners(sess, [chief_queue_runner])

        # Perform training
        # 执行分布式模型训练
        time_begin = time.time()
        print("Training begins @ %f" % time_begin)
        local_step = 0
        while True:
            # Training feed
            # 读入MNIST训练数据，默认每批次100张图片
            batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
            train_feed = {x: batch_xs, y_: batch_ys}
            _, step = sess.run([train_step, global_step], feed_dict=train_feed)
            local_step += 1
            now = time.time()
            print("%f: Worker %d: training step %d done (global step: %d)" %
                  (now, FLAGS.task_index, local_step, step))
            if step >= FLAGS.train_steps:
                break

        time_end = time.time()
        print("Training ends @ %f" % time_end)
        training_time = time_end - time_begin
        print("Training elapsed time: %f s" % training_time)

        # Validation feed
        # 读入MNIST验证数据，计算验证的交叉熵
        val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        val_xent = sess.run(cross_entropy, feed_dict=val_feed)
        print("After %d training step(s), validation cross entropy = %g" % (FLAGS.train_steps, val_xent))


if __name__ == "__main__":
    tf.app.run()
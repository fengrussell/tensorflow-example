# encoding: utf-8

import math
import tempfile
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

flags = tf.app.flags
IMAGE_PIXELS = 28

flags.DEFINE_string('data_dir', '../data/mnist', 'Directory for storing mnist data')
flags.DEFINE_integer('hidden_units', 100, 'Number of units in the hidden layer of the NN')
flags.DEFINE_integer('train_steps', 10000, 'Number of training steps to perform')
flags.DEFINE_integer('batch_size', 100, 'Training batch size ')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')

flags.DEFINE_string('ps_hosts', '127.0.0.1:50000', 'Comma-separated list of hostname:port pairs')
flags.DEFINE_string('worker_hosts', '127.0.0.1:50001,127.0.0.1:50002', 'Comma-separated list of hostname:port pairs')

flags.DEFINE_string('job_name', None, 'job name: worker or ps')
flags.DEFINE_integer('task_index', None, 'Index of task within the job')
flags.DEFINE_integer('issync', None, '是否采用分布式的同步模式，1表示同步模式，0表示异步模式')  # 没有这个参数设置，是异步模式

FLAGS = flags.FLAGS


def main(_):
    mnist = mnist_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    if FLAGS.job_name is None or FLAGS.job_name == '':
        raise ValueError('Must specify an explicit job_name !')
    else:
        print 'job_name : %s' % FLAGS.job_name

    if FLAGS.task_index is None or FLAGS.task_index == '':
        raise ValueError('Must specify an explicit task_index!')
    else:
        print 'task_index : %d' % FLAGS.task_index

    ps_spec = FLAGS.ps_host.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')

    num_worker = len(worker_spec)
    cluster = tf.train.ClusterSpec({'ps': ps_spec, 'workder': worker_spec})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    if FLAGS.job_name == 'ps':
        server.join()

    is_chief = (FLAGS.task_index == 0)
    # worker_device = '/job:worker/task%d/cpu:0' % FLAGS.task_index
    with tf.device(tf.train.replica_device_setter(cluster = cluster)):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        hid_w = tf.Variable(tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units], stddev=1.0 / IMAGE_PIXELS), 'hid_w')
        hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name='hid_b')

        sm_w = tf.Variable(tf.truncated_normal([FLAGS.hidden_units, 10], stddev=1.0 / math.sqrt(FLAGS.hidden_units)), name='sm_w')
        sm_b = tf.Variable(tf.zeros([10]), name='sm_b')

        x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
        y_ = tf.placeholder(tf.float32, [None, 10])

        hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
        hid = tf.nn.relu(hid_lin)

        y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
        cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

        opt = tf.train.AdadeltaOptimizer(FLAGS.learning_rate)

        train_step = opt.minimize(cross_entropy, global_step=global_step)

        init_op = tf.global_variables_initializer()
        train_dir = tempfile.mkdtemp()

        sv = tf.train.Supervisor(is_chief=is_chief, logidr=train_dir, init_op=init_op, recovery_wait_secs=1, global_step=global_step)

        if is_chief:
            print 'Worker %d: Initailizing session...' % FLAGS.task_index
        else:
            print 'Worker %d: Waiting for session to be initaialized...' % FLAGS.task_index

        sess = sv.prepare_or_wait_for_session(server.target)
        print 'Worker %d: Session initialization  complete.' % FLAGS.task_index

        time_begin = time.time()
        print 'Traing begins @ %f' % time_begin

        local_step = 0
        while True:
            batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
            train_feed = {x: batch_xs, y_: batch_ys}

            _, step = sess.run([train_step, global_step], feed_dict=train_feed)
            local_step += 1

            now = time.time()
            print '%f: Worker %d: traing step %d dome (global step:%d)' % (now, FLAGS.task_index, local_step, step)

            if step >= FLAGS.train_steps:
                break

        time_end = time.time()
        print 'Training ends @ %f' % time_end
        train_time = time_end - time_begin
        print 'Training elapsed time:%f s' % train_time

        val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        val_xent = sess.run(cross_entropy, feed_dict=val_feed)
        print 'After %d training step(s), validation cross entropy = %g' % (FLAGS.train_steps, val_xent)
        sess.close


if __name__ == '__main__':
    tf.app.run()
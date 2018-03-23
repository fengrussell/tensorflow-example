# !/usr/bin/env python2
# -*- coding: utf-8 -*-

# code url: http://blog.csdn.net/guotong1988/article/details/53780424

from __future__ import print_function
import time
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/mnist_data/", one_hot=True)

# cluster specification
ps = ['127.0.0.1:2222']
workers = ['127.0.0.1:2223', '127.0.0.1:2224']

cluster = tf.train.ClusterSpec({"ps": ps, "worker": workers})

# input flags
tf.app.flags.DEFINE_string('job_name', '', 'Either \'ps\' or \'worker\'')
tf.app.flags.DEFINE_integer('task_index', '0', 'Index of taks within the job')
FLAGS = tf.app.flags.FLAGS

# start a server for specific task
server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

# config
batch_size = 128
learning_rate = 0.001
training_epochs = 20
logs_path = './log/mnist'


if FLAGS.job_name == 'ps':
    server.join()
elif FLAGS.job_name == 'worker':

    # replica_device_setter没有显式的指定ps_device, 应该是可以从cluster获取, 待从源码确认。 2018-03-30
    with tf.device(tf.train.replica_device_setter(
            worker_device='/job:worker/task:%d' % FLAGS.task_index,
            cluster=cluster)):

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, shape=[None, 784], name='input-x')
            y = tf.placeholder(tf.float32, shape=[None, 10], name='input-y')

        tf.set_random_seed(1)
        with tf.name_scope('weight'):
            W = tf.Variable(tf.random_normal([784, 10]))

        with tf.name_scope('bias'):
            b = tf.Variable(tf.random_normal([10]))

        with tf.name_scope('softmax'):
            y_pred = tf.nn.softmax(tf.add(tf.matmul(x, W), b))

        with tf.name_scope('cross_entropy'):
            # this is our cost
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_pred * tf.log(y), reduction_indices=[1]))

            # specify optimizer
            with tf.name_scope('train'):
                # optimizer is an "operation" which we can execute in a session
                grad_op = tf.train.GradientDescentOptimizer(learning_rate)
                '''
                rep_op = tf.train.SyncReplicasOptimizer(grad_op,                    
                                                      replicas_to_aggregate=len(workers),
                                                      replica_id=FLAGS.task_index, 
                                                      total_num_replicas=len(workers),
                                                      use_locking=True)
                train_op = rep_op.minimize(cross_entropy, global_step=global_step)
                '''
                train_op = grad_op.minimize(cross_entropy, global_step=global_step)

            '''
            init_token_op = rep_op.get_init_tokens_op()
            chief_queue_runner = rep_op.get_chief_queue_runner()
            '''

            with tf.name_scope('Accuracy'):
                # accuracy
                correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # create a summary for our cost and accuracy
            tf.summary.scalar("cost", cross_entropy)
            tf.summary.scalar("accuracy", accuracy)

            # merge all summaries into a single "operation" which we can execute in a session
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()
            print("Variables initialized ...")

        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 global_step=global_step,
                                 init_op=init_op)

        begin_time = time.time()
        freq = 100
        with sv.prepare_or_wait_for_session(server.target) as sess:

            writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

            start_time = time.time()
            for epoch in range(training_epochs):

                batch_count = int(mnist.train.num_examples/batch_size)

                count = 0
                for i in range(batch_count):
                    batch_x, batch_y = mnist.train.next_batch(batch_size)

                    _, cost, summary, step = sess.run(
                        [train_op, cross_entropy, summary_op, global_step],
                        feed_dict={x: batch_x, y: batch_y})

                    count += 1
                    if count % freq == 0 or i + 1 == batch_count:
                        elapsed_time = time.time() - start_time
                        start_time = time.time()
                        print("Step: %d," % (step + 1),
                              " Epoch: %2d," % (epoch + 1),
                              " Batch: %3d of %3d," % (i + 1, batch_count),
                              " Cost: %.4f," % cost,
                              " AvgTime: %3.2fms" % float(elapsed_time * 1000 / freq))
                        count = 0

            print("Test-Accuracy: %2.2f" % sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
            print("Total Time: %3.2fs" % float(time.time() - begin_time))
            print("Final Cost: %.4f" % cost)

    sv.stop()
    print('done')

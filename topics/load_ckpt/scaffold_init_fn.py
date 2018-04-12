#! -*- coding: utf-8 -*-
# https://stackoverflow.com/questions/45107068/how-to-fine-tune-model-using-monitoredtrainingsession-scaffold


import sys
import tensorflow as tf
slim = tf.contrib.slim
import argparse
# import model as M
# import decoder as D

FLAGS = None


def train(_):
    vgg_19_ckpt_path='/media/data/projects/project_daphnis/pretrained_models/vgg_19.ckpt'
    train_log_dir = "/media/data/projects/project_daphnis/train_log_dir"

    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        if not tf.gfile.Exists(train_log_dir):
            tf.gfile.MakeDirs(train_log_dir)

        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            # Set up the data loading:
            image, c, p, s = \
                D.get_training_dataset_data_provider()

            image, c, p, s = \
                tf.train.batch([image, c, p, s],
                               batch_size=16)

            # Define the model:
            predictions, loss, end_points = M.model_as_in_paper(
                image, c, p, s
            )

            values_to_restore = slim.get_variables_to_restore(
                include=["vgg_19"],
                exclude=[
                    'vgg_19/conv4_3_X',
                    'vgg_19/conv4_4_X']
            )

            # Specify the optimization scheme:
            optimizer = tf.train.AdamOptimizer(learning_rate=.00001)

            # create_train_op that ensures that when we evaluate it to get the loss,
            # the update_ops are done and the gradient updates are computed.
            train_op = slim.learning.create_train_op(loss, optimizer)
        tf.summary.scalar("losses/total_loss", loss)

        # The StopAtStepHook handles stopping after running given steps.
        hooks = [tf.train.StopAtStepHook(last_step=1000000)]

        pre_train_saver = tf.train.Saver(values_to_restore)

        def load_pretrain(scaffold, sess):
            pre_train_saver.restore(sess,
                                    vgg_19_ckpt_path)

        # The MonitoredTrainingSession takes care of session initialization,
        # restoring from a checkpoint, saving to a checkpoint, and closing when done
        # or an error occurs.
        with tf.train.MonitoredTrainingSession(
                master=server.target,
                is_chief=(FLAGS.task_index == 0),
                checkpoint_dir=train_log_dir,
                hooks=hooks,
                scaffold=tf.train.Scaffold(
                    init_fn=load_pretrain,
                    summary_op=tf.summary.merge_all())) as mon_sess:

            while not mon_sess.should_stop():
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.
                # mon_sess.run handles AbortedError in case of preempted PS.
                mon_sess.run(train_op)


if __name__ == "__main__":
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.register("type", "bool", lambda v: v.lower() == "true")
        # Flags for defining the tf.train.ClusterSpec
        parser.add_argument(
            "--ps_hosts",
            type=str,
            default="",
            help="Comma-separated list of hostname:port pairs"
        )
        parser.add_argument(
            "--worker_hosts",
            type=str,
            default="",
            help="Comma-separated list of hostname:port pairs"
        )
        parser.add_argument(
            "--job_name",
            type=str,
            default="",
            help="One of 'ps', 'worker'"
        )
        # Flags for defining the tf.train.Server
        parser.add_argument(
            "--task_index",
            type=int,
            default=0,
            help="Index of task within the job"
        )
        FLAGS, unparsed = parser.parse_known_args()
        tf.app.run(main=train, argv=[sys.argv[0]] + unparsed)
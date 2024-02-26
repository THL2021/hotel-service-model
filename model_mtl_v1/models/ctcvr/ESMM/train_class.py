# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import numpy as np
import time
from tools import *
from model_fn import model_block
from tools import auc as local_auc

# -----------------------------------------------------------------------------------------------------------
tf.logging.set_verbosity(tf.logging.INFO)
FLAGS = None
linear_parent_scope = "linear"
dnn_parent_scope = "dnn"


def auc_helper(labels, preds, weights, name, is_local=False):
    auc_func = tf.metrics.auc if not is_local else local_auc.auc
    return auc_func(
        labels=tf.reshape(labels, [-1, 1]),
        predictions=tf.reshape(preds, [-1, 1]),
        weights=weights, name=name)


# -----------------------------------------------------------------------------------------------------------
def train_model_fn(data_batch, global_step):
    features, label_dict, fc_generator, params = parse_fg(data_batch, FLAGS)
    logit_dict, label_dict = model_block(features, label_dict, fc_generator, FLAGS.is_training, FLAGS.keep_prob, global_step, params)
    loss, metrics = train_loss_and_metrics_step(logit_dict, label_dict, "loss_and_metrics")
    train_op = make_training_op(loss, global_step, FLAGS.is_sequence_train)
    return loss, train_op, metrics


def focal_loss_sigmoid(labels, logits, alpha=0.5, gamma=2):
    """
    Computer focal loss for binary classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size].
      alpha: A scalar for focal loss alpha hyper-parameter. If positive samples number
      > negtive samples number, alpha < 0.5 and vice versa.
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred = tf.nn.sigmoid(logits)
    labels = tf.to_float(labels)
    focal_loss = -labels * (1 - alpha) * ((1 - y_pred) ** gamma) * tf.log(y_pred) \
                 - (1 - labels) * alpha * (y_pred ** gamma) * tf.log(1 - y_pred)
    return focal_loss


def train_loss_and_metrics_step(logit_dict, label_dict, scope):
    with tf.variable_scope(scope):
        logit_cancel = logit_dict['logit_cancel']
        logit_noroom = logit_dict['logit_noroom']
        logit_noroom_auxi = logit_dict['logit_noroom_auxi']

        label_cancel = label_dict['label_cancel']
        label_noroom = label_dict['label_noroom']
        label_noroom_all = label_dict['label_noroom_all']

        is_noroom_sample = label_dict['is_noroom_sample']
        is_cancel_sample = label_dict['is_cancel_sample']

        label_all = tf.where(label_cancel + label_noroom > 0,
                             tf.ones_like(label_cancel),
                             tf.zeros_like(label_cancel))

        with tf.name_scope('loss'):
            prob_cancel = tf.sigmoid(logit_cancel)
            loss_cancel = is_cancel_sample * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_cancel,
                                                                                     logits=logit_cancel,
                                                                                     name='loss_cancel')

            loss_cancel_fl = focal_loss_sigmoid(labels=label_cancel, logits=logit_cancel)
            auc_train_cancel = auc_helper(tf.reshape(label_cancel, [-1, 1]),
                                          tf.reshape(prob_cancel, [-1, 1]),
                                          tf.reshape(tf.ones_like(label_cancel), [-1, 1]),
                                          'auc-train-cancel',
                                          is_local=True)

            prob_noroom = tf.sigmoid(logit_noroom)
            loss_noroom = is_noroom_sample * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_noroom,
                                                                                     logits=logit_noroom,
                                                                                     name='loss_noroom')
            loss_noroom_fl = focal_loss_sigmoid(labels=label_noroom, logits=logit_noroom)
            auc_train_noroom = auc_helper(tf.reshape(label_noroom, [-1, 1]),
                                          tf.reshape(prob_noroom, [-1, 1]),
                                          tf.reshape(tf.ones_like(label_noroom), [-1, 1]),
                                          'auc-train-noroom',
                                          is_local=True)

            loss_noroom_auxi = is_noroom_sample * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_noroom,
                                                                                          logits=logit_noroom_auxi,
                                                                                          name='loss_noroom_auxi')
            loss_noroom_auxi_fl = focal_loss_sigmoid(labels=label_noroom, logits=logit_noroom_auxi)

            loss_cancel_v = tf.reduce_mean(loss_cancel)
            loss_noroom_v = tf.reduce_mean(loss_noroom)
            loss = tf.reduce_mean(loss_cancel) + tf.reduce_mean(loss_noroom) + tf.reduce_mean(loss_noroom_auxi)

            tf.add_to_collection("losses", loss)
            losses = tf.get_collection('losses')
            tf.logging.info("consider losses: {}".format(losses))
            loss_total = tf.add_n(losses)
        with tf.name_scope("metrics"):
            metrics = [loss_cancel_v, loss_noroom_v, auc_train_cancel, auc_train_noroom]
        return loss_total, metrics


def train_step(loss, train_op, train_metrics, step, global_step, sess, step_name):
    loss_cancel, loss_noroom, auc_train_cancel, auc_train_noroom = train_metrics
    _, loss_val, global_step_val, loss_cancel_val, loss_noroom_val, auc_train_cancel_val, auc_train_noroom_val = sess.run(
        [train_op, loss, global_step, loss_cancel, loss_noroom, auc_train_cancel, auc_train_noroom])
    tf.logging.info(
        'Step {} global_step {} max_step {} result: loss_total: {} loss_cancel: {} loss_noroom: {} '
        'auc_train_cancel: {} auc_train_noroom: {}'.format(
            step, global_step_val, FLAGS.max_train_step,
            loss_val, loss_cancel_val, loss_noroom_val, auc_train_cancel_val, auc_train_noroom_val))
    return global_step_val


def train(worker_count, task_index, cluster, is_chief, target):
    worker_device = "/job:worker/task:%d/cpu:%d" % (task_index, 0)
    tf.logging.info("worker_device = %s" % worker_device)

    model_dir_restore = FLAGS.model_dir_restore
    model_dir = FLAGS.model_dir
    batch_size = FLAGS.batch_size
    checkpointDir = FLAGS.checkpointDir
    buckets = FLAGS.buckets
    model_dir = os.path.join(checkpointDir, model_dir)
    model_dir_restore = os.path.join(checkpointDir, model_dir_restore)
    tf.logging.info(
        "buckets:{} checkpointDir:{} checkpointDir_restore:{}".format(buckets, model_dir, model_dir_restore))
    # -----------------------------------------------------------------------------------------------
    tf.logging.info("loading input...")
    train_file = FLAGS.train_tables.split(',')

    with tf.device(worker_device):
        train_dataset = input_fn(train_file, batch_size, 'train', is_sequence_train=FLAGS.is_sequence_train,
                                 slice_count=worker_count, slice_id=task_index, epochs=2)
        train_iterator = train_dataset.make_one_shot_iterator()
    tf.logging.info("finished loading input...")

    available_worker_device = "/job:worker/task:%d" % (task_index)
    with tf.device(tf.train.replica_device_setter(worker_device=available_worker_device, cluster=cluster)):
        global_step = tf.Variable(0, name="global_step", trainable=False)
        loss, train_op, train_metrics = train_model_fn(train_iterator, global_step)

    tf.logging.info("start training")
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    for var in bn_moving_vars:
        tf.add_to_collection(tf.GraphKeys.TRAINABLE_VARIABLES, var)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, var)

    hooks = [tf.train.StopAtStepHook(last_step=FLAGS.max_train_step)]
    hooks_for_chief = [
        tf.train.CheckpointSaverHook(
            checkpoint_dir=model_dir,
            save_secs=FLAGS.save_time,
            saver=tf.train.Saver(name='chief_saver'))
    ]

    if FLAGS.is_restore:
        ckpt_state = tf.train.get_checkpoint_state(model_dir_restore)
        if not (ckpt_state and tf.train.checkpoint_exists(ckpt_state.model_checkpoint_path)):
            tf.logging.info("restore path error!!!")
            raise ValueError
    else:
        model_dir_restore = None

    step = 0

    with tf.train.MonitoredTrainingSession(checkpoint_dir=model_dir_restore,
                                           master=target,
                                           is_chief=is_chief,
                                           config=sess_config,
                                           save_checkpoint_secs=None,
                                           hooks=hooks,
                                           chief_only_hooks=hooks_for_chief) as sess:

        chief_is_end = False
        sess_is_end = False
        while (not sess_is_end) and (not sess.should_stop()):
            if not chief_is_end:
                try:
                    step += 1
                    global_step_val = train_step(loss, train_op, train_metrics, step, global_step, sess, "step")
                except tf.errors.OutOfRangeError as e:
                    if is_chief:
                        tf.logging.info("chief node end...")
                        chief_is_end = True
                        tf.logging.info("waiting all worker nodes to be end")
                        last_step = global_step_val
                    else:
                        tf.logging.info("worker node end...")
                        break
            else:
                while 1:
                    time.sleep(60)
                    tf.logging.info("waiting all worker nodes to be end")
                    global_step_val = sess.run(global_step)
                    if global_step_val > last_step:
                        last_step = global_step_val
                    else:
                        tf.logging.info("all worker nodes end. chief node is finished")
                        sess_is_end = True
                        break
    tf.logging.info("%d steps finished." % step)


def distribute_train(FLAGS_):
    global FLAGS
    FLAGS = FLAGS_

    tf.logging.info("job name = %s" % FLAGS.job_name)
    tf.logging.info("task index = %d" % FLAGS.task_index)
    is_chief = FLAGS.task_index == 0
    ps_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})
    worker_count = len(worker_spec)
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        server.join()
    train(worker_count=worker_count, task_index=FLAGS.task_index, cluster=cluster, is_chief=is_chief,
          target=server.target)

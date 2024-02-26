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


def get_ghm_weight(predict, target, valid_mask, bins=10, alpha=0.75,
                   dtype=tf.float32, name='GHM_weight'):
    """ Get gradient Harmonized Weights.
    This is an implementation of the GHM ghm_weights described
    in https://arxiv.org/abs/1811.05181.
    Args:
        predict:
            The prediction of categories branch, [0, 1].
            -shape [batch_num, category_num].
        target:
            The target of categories branch, {0, 1}.
            -shape [batch_num, category_num].
        valid_mask:
            The valid mask, is 0 when the sample is ignored, {0, 1}.
            -shape [batch_num, category_num].
        bins:
            The number of bins for region approximation.
        alpha:
            The moving average parameter.
        dtype:
            The dtype for all operations.

    Returns:
        weights:
            The beta value of each sample described in paper.
    """
    with tf.variable_scope(name):
        _edges = [x / bins for x in range(bins + 1)]
        _edges[-1] += 1e-6
        edges = tf.constant(_edges, dtype=dtype)

        _shape = predict.get_shape().as_list()

        _init_statistics = 2000 / bins
        statistics = tf.get_variable(
            name='statistics', shape=[bins], dtype=dtype, trainable=False,
            initializer=tf.constant_initializer(_init_statistics, dtype=dtype))

        _b_valid = valid_mask > 0
        total = tf.maximum(tf.reduce_sum(tf.cast(_b_valid, dtype=dtype)), 1)

        gradients = tf.abs(predict - target)

        # Calculate new statics and new weights
        w_list = []
        s_list = []
        for i in range(bins):
            inds = (
                           gradients >= edges[i]) & (gradients < edges[i + 1]) & _b_valid
            # number of examples lying in bin, same as R in paper.
            num_in_bin = tf.reduce_sum(tf.cast(inds, dtype=dtype))
            statistics_i = alpha * statistics[i] + (1 - alpha) * num_in_bin
            gradient_density = statistics_i * bins
            update_weights = total / gradient_density
            weights_i = tf.where(
                inds,
                x=tf.ones_like(predict) * update_weights,
                y=tf.zeros_like(predict))
            w_list.append(weights_i)
            s_list.append(statistics_i)

        weights = tf.add_n(w_list)
        new_statistics = tf.stack(s_list)

        # Avoid the tiny value in statistics
        new_statistics = tf.maximum(new_statistics, _init_statistics)
        # Update statistics
        statistics_updated_op = statistics.assign(new_statistics)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, statistics_updated_op)

    return weights


def focal_loss_sigmoid(labels, logits, alpha=0.25, gamma=2):
    """
    Computer focal loss for binary classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size].
      alpha: A scalar for focal loss alpha hyper-parameter. If positive samples number
      > negtive samples number, alpha > 0.5 and vice versa.
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    eps = 1e-6
    y_pred = tf.nn.sigmoid(logits)
    y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
    labels = tf.to_float(labels)
    focal_loss = -labels * ((1 - y_pred) ** gamma) * tf.log(y_pred) \
                 - (1 - labels) * (y_pred ** gamma) * tf.log(1 - y_pred)
    return focal_loss


def cyclical_focal_loss(labels, logits, alpha=0.25, gamma=2):
    eps = 1e-6
    y_pred = tf.nn.sigmoid(logits)
    y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
    labels = tf.to_float(labels)
    cf_loss = -labels * ((1 + y_pred) ** gamma) * tf.log(y_pred) \
              - (1 - labels) * ((2 - y_pred) ** gamma) * tf.log(1 - y_pred)
    return cf_loss


# -----------------------------------------------------------------------------------------------------------
def train_model_fn(data_batch, global_step):
    features, label_dict, fc_generator, params = parse_fg(data_batch, FLAGS)
    logit_dict, label_dict = model_block(features, label_dict, fc_generator, FLAGS.is_training, FLAGS.keep_prob, global_step, params)
    loss, metrics = train_loss_and_metrics_step(logit_dict, label_dict, global_step, "loss_and_metrics")
    train_op = make_training_op(loss, global_step, FLAGS.is_sequence_train)
    return loss, train_op, metrics


def train_loss_and_metrics_step(logit_dict, label_dict, global_step, scope):
    with tf.variable_scope(scope):
        logit_noroom = logit_dict['logit_noroom']
        logit_cancel = logit_dict['logit_cancel']
        label = label_dict['click']
        label_cancel = label_dict['pay']

        # label_cancel = tf.where(label_cancel + label > 0, tf.ones_like(label), tf.zeros_like(label))

        is_cancel_sample = logit_dict['is_cancel_sample']
        is_noroom_sample = logit_dict['is_noroom_sample']

        is_cancel_sample = tf.where((1 - label) + label_cancel > 0, tf.ones_like(label), tf.zeros_like(label))

        prob = tf.sigmoid(logit_noroom)
        prob_cancel = tf.sigmoid(logit_cancel)

        target = tf.concat([1 - label, label], axis=1)
        predict = tf.concat([1 - prob, prob], axis=1)
        valid_mask = tf.ones_like(target)

        with tf.name_scope('loss'):
            # ghm_weight = get_ghm_weight(predict, target, valid_mask, bins=10, alpha=0.75,
            #                             dtype=tf.float32, name='GHM_weight')
            # lc_loss = focal_loss_sigmoid(labels=label, logits=logit)
            # hc_loss = cyclical_focal_loss(labels=label, logits=logit)
            #
            # ce_loss = tf.nn.weighted_cross_entropy_with_logits(targets=label, logits=logit, pos_weight=5, name='loss')
            # ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit, name='loss')
            # w1 = tf.Variable(1.0 - global_step / 6000.0, name="loss_w1", trainable=False)
            # w2 = tf.Variable((global_step / 6000.0 - 1.0) / (3.1 - 1), name="loss_w2", trainable=False)
            # w = tf.cond(global_step < 6000, lambda: w1, lambda: w2)
            # w = tf.cond(global_step < 6000,
            #             lambda: tf.subtract(1.0, tf.div(tf.to_float(global_step), 6000.0)),
            #             lambda: tf.div(tf.subtract(tf.div(tf.to_float(global_step), 6000.0), 1.0), (3.1 - 1)))
            #
            # loss = w * ce_loss + (1 - w) * lc_loss

            # loss = ghm_weight * tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit, name='loss')
            # loss_noroom = tf.nn.weighted_cross_entropy_with_logits(targets=label, logits=logit_noroom, pos_weight=5, name='loss')
            # focal_loss = focal_loss_sigmoid(labels=label, logits=logit)

            loss_noroom = is_noroom_sample * tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit_noroom, name='loss_noroom')
            loss_cancel = is_cancel_sample * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_cancel, logits=logit_cancel, name='loss_cancel')

            loss_noroom = tf.reduce_mean(loss_noroom)
            loss_cancel = tf.reduce_mean(loss_cancel)

            loss = loss_noroom + 0.1 * loss_cancel
            tf.summary.scalar("loss", loss)
            tf.summary.scalar("loss_noroom", loss_noroom)
            tf.summary.scalar("loss_cancel", loss_cancel)

            auc_train = auc_helper(tf.reshape(label, [-1, 1]),
                                   tf.reshape(prob, [-1, 1]),
                                   tf.reshape(tf.ones_like(label), [-1, 1]),
                                   'auc-train',
                                   is_local=True)
            auc_train_cancel = auc_helper(tf.reshape(label_cancel, [-1, 1]),
                                          tf.reshape(prob_cancel, [-1, 1]),
                                          tf.reshape(tf.ones_like(label_cancel), [-1, 1]),
                                          'auc-train-cancel',
                                          is_local=True)
            tf.summary.scalar("auc_train", auc_train)
            tf.summary.scalar("auc_train_cancel", auc_train_cancel)

            tf.add_to_collection("losses", loss)
            losses = tf.get_collection('losses')
            tf.logging.info("consider losses: {}".format(losses))
            loss_total = tf.add_n(losses)

        merge_summary = tf.summary.merge_all()
        with tf.name_scope("metrics"):
            metrics = [loss, loss_noroom, loss_cancel, auc_train, auc_train_cancel, merge_summary]
        return loss_total, metrics


def train_step(loss, train_op, train_metrics, step, global_step, sess, step_name):
    loss, loss_noroom, loss_cancel, auc_train, auc_train_cancel, merge_summary = train_metrics
    _, global_step_val, loss_, loss_noroom_, loss_cancel_, auc_train_, auc_train_cancel_, merge_summary_ = sess.run(
        [train_op, global_step, loss, loss_noroom, loss_cancel, auc_train, auc_train_cancel, merge_summary])
    tf.logging.info(
        '[loss, loss_noroom, loss_cancel, auc_train, auc_train_cancel] at step {} global_step {}: [{}, {}, {}, {}, {}]'.format(
            step, global_step_val,
            loss_, loss_noroom_, loss_cancel_, auc_train_, auc_train_cancel_))
    return global_step_val, merge_summary_


def train(worker_count, task_index, cluster, is_chief, target):
    worker_device = "/job:worker/task:%d/cpu:%d" % (task_index, 0)
    tf.logging.info("worker_deivce = %s" % worker_device)

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
        train_writer = tf.summary.FileWriter(model_dir, sess.graph)
        while (not sess_is_end) and (not sess.should_stop()):
            if not chief_is_end:
                try:
                    step += 1
                    global_step_val, merge_summary_ = train_step(loss, train_op, train_metrics, step, global_step, sess, "step")
                    train_writer.add_summary(merge_summary_, global_step_val)
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

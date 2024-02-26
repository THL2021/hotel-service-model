# -*- coding: utf-8 -*-
import os
from model_fn import model_block
from tools import *
import time
from collections import defaultdict

# -----------------------------------------------------------------------------------------------------------
tf.logging.set_verbosity(tf.logging.INFO)
FLAGS = None
linear_parent_scope = "linear"
dnn_parent_scope = "dnn"


# -----------------------------------------------------------------------------------------------------------
def dev_loss_and_metrics_step(logit_dict, label_dict, scope):
    with tf.variable_scope(scope):
        with tf.name_scope("metrics"):
            logit_cancel = logit_dict['logit_cancel']
            logit_noroom = logit_dict['logit_noroom']

            label_cancel = label_dict['label_cancel']
            label_noroom = label_dict['label_noroom']

            is_noroom_sample = label_dict['is_noroom_sample']
            is_cancel_sample = label_dict['is_cancel_sample']

            prob_cancel = tf.sigmoid(logit_cancel)
            loss_cancel = is_cancel_sample * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_cancel,
                                                                                     logits=logit_cancel,
                                                                                     name='loss_cancel')

            prob_noroom = tf.sigmoid(logit_noroom)
            loss_noroom = is_noroom_sample * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_noroom,
                                                                                     logits=logit_noroom,
                                                                                     name='loss_noroom')

            loss_cancel = tf.reduce_mean(loss_cancel)
            loss_noroom = tf.reduce_mean(loss_noroom)
            loss = tf.reduce_mean(0.2 * loss_cancel + 0.8 * loss_noroom)

            loss = [loss, loss_cancel, loss_noroom]
            metrics = [label_cancel, label_noroom, prob_cancel, prob_noroom]

        return loss, metrics


def dev_model_fn(data_batch, global_step):
    features, label_dict, fc_generator, params = parse_fg(data_batch, FLAGS)
    logit_dict, label_dict = model_block(features, label_dict, fc_generator, FLAGS.is_training, FLAGS.keep_prob, tf.constant(10000, dtype=tf.int32), params)
    loss, metrics = dev_loss_and_metrics_step(logit_dict, label_dict, "loss_and_metrics")
    return loss, None, metrics


def dev_step(loss, metrics, step, sess, step_name):
    label_cancel, label_noroom, prob_cancel, prob_noroom = metrics
    loss, loss_cancel, loss_noroom = loss

    labels_cancel = []
    labels_noroom = []

    preds_cancel = []
    preds_noroom = []

    losses = []
    losses_cancel = []
    losses_noroom = []

    dev_num_batches = FLAGS.dev_total // FLAGS.batch_size + 1
    for batch_ in range(1, dev_num_batches + 1):
        tf.logging.info("batch: {}, at {}".format(batch_, step))
        try:
            label_cancel_, label_noroom_, prob_cancel_, prob_noroom_, loss_, loss_cancel_, loss_noroom_ = sess.run([
                label_cancel, label_noroom,
                prob_cancel, prob_noroom,
                loss, loss_cancel, loss_noroom])
            losses.append(loss_)
            losses_cancel.append(loss_cancel_)
            losses_noroom.append(loss_noroom_)
            for label_cancel_val, label_noroom_val, prob_cancel_val, prob_noroom_val in zip(
                    label_cancel_, label_noroom_, prob_cancel_, prob_noroom_):
                labels_cancel.append(label_cancel_val[0])
                preds_cancel.append(prob_cancel_val[0])
                if label_cancel_val[0] < 1:
                    labels_noroom.append(label_noroom_val[0])
                    preds_noroom.append(prob_noroom_val[0])
        except tf.errors.OutOfRangeError as e:
            tf.logging.info("All sample is traversed, dev end!")
            break

    labels_cancel = np.array(labels_cancel)
    preds_cancel = np.array(preds_cancel)
    auc_cancel = roc_auc_score(labels_cancel, preds_cancel)

    labels_noroom = np.array(labels_noroom)
    preds_noroom = np.array(preds_noroom)
    auc_noroom = roc_auc_score(labels_noroom, preds_noroom)

    avg_loss = np.average(np.array(losses))
    avg_loss_cancel = np.average(np.array(losses_cancel))
    avg_loss_noroom = np.average(np.array(losses_noroom))

    positive_n_cancel = int(np.sum(labels_cancel))
    positive_n_noroom = int(np.sum(labels_noroom))
    all_n = len(labels_cancel)

    print(
        "all_n: {} positive_n_cancel: {} positive_n_noroom: {} auc_cancel: {} auc_noroom: {} avg_loss: {} "
        "avg_loss_cancel: {} avg_loss_noroom: {} at {}".format(
            all_n,
            positive_n_cancel,
            positive_n_noroom,
            auc_cancel,
            auc_noroom,
            avg_loss,
            avg_loss_cancel,
            avg_loss_noroom,
            step))


def dev(FLAGS_):
    global FLAGS
    FLAGS = FLAGS_

    model_dir = FLAGS.model_dir
    batch_size = FLAGS.batch_size
    checkpointDir = FLAGS.checkpointDir
    buckets = FLAGS.buckets
    model_dir = os.path.join(checkpointDir, model_dir)
    print("buckets:{} checkpointDir:{}".format(buckets, model_dir))
    # -----------------------------------------------------------------------------------------------
    tf.logging.info("loading input...")
    dev_file = FLAGS.dev_tables.split(',')
    dev_dataset = input_fn_normal(dev_file, batch_size, 'dev')
    dev_iterator = dev_dataset.make_one_shot_iterator()
    tf.logging.info("finished loading input...")
    # -----------------------------------------------------------------------------------------------
    loss, _, metrics = dev_model_fn(dev_iterator, None)
    # -----------------------------------------------------------------------------------------------
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 log_device_placement=False)

    saver = tf.train.Saver()
    with tf.Session(config=sess_config) as sess:
        step = -1
        time_before = time.time()
        while 1:
            if time.time() - time_before > 1800 * 3:
                print('no model update and eval finish...')
                break
            ckpt_state = tf.train.get_checkpoint_state(model_dir)
            if ckpt_state is None:
                continue
            final_path = str(ckpt_state.model_checkpoint_path)
            now_step = int(final_path.split('-')[-1])
            if now_step > step:
                time_before = time.time()
                time.sleep(30)
                step = now_step
                print('test checkpoint: {}'.format(ckpt_state.model_checkpoint_path))
                saver.restore(sess, ckpt_state.model_checkpoint_path)
                dev_step(loss, metrics, step, sess, "dev_step")

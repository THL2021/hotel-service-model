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
            label = label_dict['click']
            logit = logit_dict['logit']
            prob = tf.sigmoid(logit)

            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit, name='loss')
            loss = tf.reduce_mean(loss)

            metrics = [label, prob]
        return loss, metrics


def dev_model_fn(data_batch, global_step):
    features, label_dict, fc_generator, params = parse_fg(data_batch, FLAGS)
    logit_dict, label_dict = model_block(features, label_dict, fc_generator, FLAGS.is_training, FLAGS.keep_prob, params)
    loss, metrics = dev_loss_and_metrics_step(logit_dict, label_dict, "loss_and_metrics")
    return loss, None, metrics


def dev_step(loss, metrics, step, sess, step_name):
    label, prob = metrics
    labels = []
    preds = []
    losses = []

    dev_num_batches = FLAGS.dev_total // FLAGS.batch_size + 1
    for batch_ in range(1, dev_num_batches + 1):
        tf.logging.info("batch: {}, at {}".format(batch_, step))
        try:
            label_, prob_, loss_ = sess.run([label, prob, loss])
            losses.append(loss_)
            for label_val, prob_val in zip(label_, prob_):
                labels.append(label_val[0])
                preds.append(prob_val[0])
        except tf.errors.OutOfRangeError as e:
            tf.logging.info("All sample is traversed, dev end!")
            break

    labels = np.array(labels)
    preds = np.array(preds)

    auc = roc_auc_score(labels, preds)
    avg_loss = np.average(np.array(losses))

    click_n = int(np.sum(labels))
    all_n = len(labels)

    print(
        "click_n: {} all_n: {} auc: {} avg_loss: {} at {}".format(
            click_n,
            all_n,
            auc,
            avg_loss,
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

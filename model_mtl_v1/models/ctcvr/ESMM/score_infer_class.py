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
        label = label_dict['label_noroom']
        logit = logit_dict['logit_noroom']
        prob = tf.sigmoid(logit)

        with tf.name_scope("metrics"):
            metrics = [label, prob]
        return None, metrics


def dev_model_fn(data_batch, global_step):
    features, label_dict, fc_generator, params = parse_fg(data_batch, FLAGS)
    logit_dict, label_dict = model_block(features, label_dict, fc_generator, FLAGS.is_training, FLAGS.keep_prob, tf.constant(10000, dtype=tf.int32), params)
    loss, metrics = dev_loss_and_metrics_step(logit_dict, label_dict, "loss_and_metrics")
    with tf.name_scope("auc_flag"):
        tid = tf.sparse_tensor_to_dense(features["content_id"], "unknown")
        metrics.append(tid)

    return loss, None, metrics


def dev_step(loss, metrics, step, sess, step_name):
    label, prob, tid = metrics
    outputs = []
    cnt = 0
    sep = '\x1D'
    try:
        while 1:
            cnt += 1
            tf.logging.info("batch: {}, at {}".format(cnt, step))
            label_val, prob_val, tid_val = sess.run([label, prob, tid])
            for _label_val, _prob_val, _tid_val in zip(label_val, prob_val, tid_val):
                tid_str = str(_tid_val[0])
                label_str = int(_label_val[0])
                prob_str = float(_prob_val[0])
                outputs.append((tid_str, label_str, prob_str))

    except tf.errors.OutOfRangeError:
        tf.logging.info("inference finished at {}".format(cnt))

    output_file = FLAGS.output_tables
    writer = tf.python_io.TableWriter(output_file)
    print("writing data to {}, all records: {}".format(output_file, len(outputs)))
    print("writing schema: {}".format(outputs[0]))
    records = writer.write(outputs, indices=[0, 1, 2])
    writer.close()
    print("writing data finished")


def score_infer(FLAGS_):
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
    dev_dataset = input_fn_normal(dev_file, batch_size, 'infer')
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
        while 1:
            ckpt_state = tf.train.get_checkpoint_state(model_dir)
            if ckpt_state is not None:
                break
            else:
                time.sleep(10)
        final_path = str(ckpt_state.model_checkpoint_path)
        now_step = int(final_path.split('-')[-1])
        if now_step > step:
            step = now_step
            print('infer checkpoint: {}'.format(ckpt_state.model_checkpoint_path))
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            dev_step(loss, metrics, step, sess, "dev_step")

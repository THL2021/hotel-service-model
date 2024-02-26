# -*- coding: utf-8 -*-
import tensorflow as tf
import os
from model_fn import model_block
from tools import *

# -----------------------------------------------------------------------------------------------------------
tf.logging.set_verbosity(tf.logging.INFO)
FLAGS = None
linear_parent_scope = "linear"
dnn_parent_scope = "dnn"


# -----------------------------------------------------------------------------------------------------------
def transport_model_fn(data_batch, global_step):
    features, label_dict, fc_generator, params = parse_fg(data_batch, FLAGS)
    # model
    logit_dict, label_dict = model_block(features, label_dict, fc_generator, FLAGS.is_training, FLAGS.keep_prob, tf.constant(10000, dtype=tf.int32), params)

    logit_ctr = logit_dict['ctr']
    logit_cvr = logit_dict['cvr']

    ctr_ratio = logit_dict['ctr_ratio']
    cvr_ratio = logit_dict['cvr_ratio']

    with tf.variable_scope('predict'):
        # for rank service 2.0
        p_logit = tf.pow(tf.sigmoid(logit_ctr), 1.0 + ctr_ratio) * tf.pow(tf.sigmoid(logit_cvr), 1.0 + cvr_ratio)
        predict_score = tf.identity(p_logit, name="rank_predict")
    return None, None, None


def transport(FLAGS_):
    global FLAGS
    FLAGS = FLAGS_

    tf.get_default_graph().set_shape_optimize(False)
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
    # -----------------------------------------------------------------------------------------------
    transport_model_fn(dev_iterator, None)
    # -----------------------------------------------------------------------------------------------
    tf.logging.info('started building graph...')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt_state = tf.train.get_checkpoint_state(model_dir)
        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
        saver.restore(sess, ckpt_state.model_checkpoint_path)
        tf.logging.info('finished building graph...')
        transport_op(sess, saver, ckpt_state, FLAGS)

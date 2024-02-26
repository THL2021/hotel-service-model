# -*- coding: utf-8 -*-
import tensorflow as tf
from tools import *

linear_parent_scope = "linear"
dnn_parent_scope = "dnn"


def model_block(features, label_dict, fc_generator, is_training, keep_prob, params):
    # parse params
    black_list = params['black_list'] if 'black_list' in params else ""
    ########################################################
    # dnn
    tf.logging.info("building features...")
    outputs_dict = fc_generator.get_output_dict(features, black_list)

    tf.logging.info("finished build features:")
    for key in outputs_dict:
        tf.logging.info(key)
        tf.logging.info(outputs_dict[key])

    tf.logging.info("building features:")
    feats = []
    filter_names = []
    for key in outputs_dict:
        if key not in filter_names:
            tf.logging.info(key)
            tf.logging.info(outputs_dict[key])
            feats.append((key, outputs_dict[key]))

    feats = [feat for _, feat in sorted(feats, key=lambda x: x[0])]

    tf.logging.info("Total number of features: {}".format(len(feats)))

    activation_fn = tf.nn.relu

    # def mlp_inference(input, is_training, reuse, scope):
    #     with tf.variable_scope(scope, reuse=reuse):
    #         input = layers.fully_connected(input, 256, activation_fn=None, scope='ffn_1',
    #                                        variables_collections=[dnn_parent_scope])
    #
    #         input = layers.batch_norm(input, is_training=is_training, activation_fn=activation_fn,
    #                                   variables_collections=[dnn_parent_scope])
    #
    #         logit = layers.fully_connected(input, 1, activation_fn=None, scope='ffn_2',
    #                                        variables_collections=[dnn_parent_scope])
    #
    #         return logit

    drop_rate = 0.3
    with tf.variable_scope('logit'):
        input = feats

        input = tf.concat(input, axis=1)

        input = layers.batch_norm(input, is_training=is_training, activation_fn=None,
                                  variables_collections=[dnn_parent_scope])

        input = layers.fully_connected(input, 256, activation_fn=None, scope='ffn_1',
                                       variables_collections=[dnn_parent_scope])

        input = layers.batch_norm(input, is_training=is_training, activation_fn=activation_fn,
                                  variables_collections=[dnn_parent_scope])

        input = layers.dropout(input, keep_prob=1.0 - drop_rate, is_training=is_training)

        input = layers.fully_connected(input, 128, activation_fn=None, scope='ffn_2',
                                       variables_collections=[dnn_parent_scope])

        input = layers.batch_norm(input, is_training=is_training, activation_fn=activation_fn,
                                  variables_collections=[dnn_parent_scope])

        input = layers.dropout(input, keep_prob=1.0 - drop_rate, is_training=is_training)

        input = layers.fully_connected(input, 64, activation_fn=None, scope='ffn_3',
                                       variables_collections=[dnn_parent_scope])

        input = layers.batch_norm(input, is_training=is_training, activation_fn=activation_fn,
                                  variables_collections=[dnn_parent_scope])

        logit = layers.fully_connected(input, 1, activation_fn=None, scope='logit',
                                       variables_collections=[dnn_parent_scope])

    logit_dict = {}
    logit_dict['logit'] = logit

    label_click = label_dict['click']
    label_click = tf.cast(tf.equal(label_click, '1'), tf.float32)
    label_click = tf.reshape(label_click, [-1, 1])
    label_dict['click'] = label_click

    # label_pay = label_dict['pay']
    # label_pay = tf.cast(tf.equal(label_pay, '1'), tf.float32)
    # label_pay = tf.reshape(label_pay, [-1, 1])
    # label_dict['pay'] = label_pay
    return logit_dict, label_dict

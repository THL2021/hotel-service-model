# -*- coding: utf-8 -*-
import tensorflow as tf
from tools import *

linear_parent_scope = "linear"
dnn_parent_scope = "dnn"


def model_block(features, label_dict, fc_generator, is_training, keep_prob, global_step, params):
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

    # cancel_feat_list = ["shid", "hid", "seller_id", "star", "brand", "jk_type", "domestic", "province", "city",
    #                     "district", "geohash5", "geohash6", "is_credit_htl", "business", "types", "hotel_types",
    #                     "hotel_tags", "center_distance_range", "rooms", "business_num", "pic_num", "pic_room_num",
    #                     "description_length", "sr_score", "seller_type_desc", "sub_seller_type_desc"]

    id_fea_list = ["shid", "hid", "seller_id", "star", "brand", "jk_type", "domestic", "province", "city", "district",
                   "geohash5", "geohash6", "is_credit_htl", "business", "types", "hotel_types", "hotel_tags",
                   "seller_type_desc", "sub_seller_type_desc"]

    noroom_feat_list = ["shid_dp1", "hid_dp1", "seller_id_dp1", "star_dp1", "brand_dp1", "jk_type_dp1", "domestic_dp1",
                        "province_dp1", "city_dp1", "district_dp1", "geohash5_dp1", "geohash6_dp1", "is_credit_htl_dp1",
                        "business_dp1", "types_dp1", "hotel_types_dp1", "hotel_tags_dp1",
                        "seller_type_desc_dp1", "sub_seller_type_desc_dp1"]

    cancel_feat_list = ["shid_dp2", "hid_dp2", "seller_id_dp2", "star_dp2", "brand_dp2", "jk_type_dp2", "domestic_dp2",
                        "province_dp2", "city_dp2", "district_dp2", "geohash5_dp2", "geohash6_dp2", "is_credit_htl_dp2",
                        "business_dp2", "types_dp2", "hotel_types_dp2", "hotel_tags_dp2",
                        "seller_type_desc_dp2", "sub_seller_type_desc_dp2"]

    tf.logging.info("building features:")
    feats = []
    filter_names = ['label_noromm_noord', 'is_noroom_sample', 'is_cancel_sample'] + noroom_feat_list + cancel_feat_list

    for key in outputs_dict:
        if key not in filter_names:
            tf.logging.info(key)
            tf.logging.info(outputs_dict[key])
            feats.append((key, outputs_dict[key]))

    feats = [feat for _, feat in sorted(feats, key=lambda x: x[0])]

    tf.logging.info("Total number of features: {}".format(len(feats)))

    # cancel_feat_list = id_fea_list
    cancel_feats = []
    for key in outputs_dict:
        if key in cancel_feat_list:
            cancel_feats.append((key, outputs_dict[key]))

    cancel_feats = [feat for _, feat in sorted(cancel_feats, key=lambda x: x[0])]

    noroom_feat_list = id_fea_list
    noroom_auxi_feats = []
    for key in outputs_dict:
        if key in noroom_feat_list:
            noroom_auxi_feats.append((key, outputs_dict[key]))

    noroom_auxi_feats = [feat for _, feat in sorted(noroom_auxi_feats, key=lambda x: x[0])]

    tf.logging.info("Total number of cancel features: {}".format(len(cancel_feats)))

    def mlp_inference(input, repr_emb, activation_fn, drop_rate, is_training, reuse, scope):
        with tf.variable_scope(scope, reuse=reuse):
            input = layers.fully_connected(input, 256, activation_fn=None, scope='ffn_1',
                                           variables_collections=[dnn_parent_scope])

            input = layers.batch_norm(input, is_training=is_training, activation_fn=activation_fn,
                                      variables_collections=[dnn_parent_scope])

            input = layers.dropout(input, keep_prob=1.0 - drop_rate, is_training=is_training)

            input = layers.fully_connected(input, 128, activation_fn=None, scope='ffn_2',
                                           variables_collections=[dnn_parent_scope])

            input = layers.batch_norm(input, is_training=is_training, activation_fn=activation_fn,
                                      variables_collections=[dnn_parent_scope])

            # repr_emb = tf.cond(global_step < 5000, lambda: tf.zeros_like(repr_emb), lambda: repr_emb)

            input = tf.concat([input, repr_emb], axis=1)

            input = layers.dropout(input, keep_prob=1.0 - drop_rate, is_training=is_training)

            input = layers.fully_connected(input, 64, activation_fn=None, scope='ffn_3',
                                           variables_collections=[dnn_parent_scope])

            input = layers.batch_norm(input, is_training=is_training, activation_fn=activation_fn,
                                      variables_collections=[dnn_parent_scope])

            logit = layers.fully_connected(input, 1, activation_fn=None, scope='logit',
                                           variables_collections=[dnn_parent_scope])

            return logit

    def auxiliary_net(input, activation_fn, drop_rate, is_training, reuse, scope):
        with tf.variable_scope(scope, reuse=reuse):
            input = layers.fully_connected(input, 128, activation_fn=None, scope='ffn_1',
                                           variables_collections=[dnn_parent_scope])

            input = layers.batch_norm(input, is_training=is_training, activation_fn=activation_fn,
                                      variables_collections=[dnn_parent_scope])

            input = layers.dropout(input, keep_prob=1.0 - drop_rate, is_training=is_training)

            input = layers.fully_connected(input, 64, activation_fn=None, scope='ffn_2',
                                           variables_collections=[dnn_parent_scope])

            input = layers.batch_norm(input, is_training=is_training, activation_fn=activation_fn,
                                      variables_collections=[dnn_parent_scope])

            input = layers.dropout(input, keep_prob=1.0 - drop_rate, is_training=is_training)

            input = layers.fully_connected(input, 32, activation_fn=None, scope='ffn_3',
                                           variables_collections=[dnn_parent_scope])

            input = layers.batch_norm(input, is_training=is_training, activation_fn=activation_fn,
                                      variables_collections=[dnn_parent_scope])

            repr_emb = input

            logit = layers.fully_connected(input, 1, activation_fn=None, scope='logit',
                                           variables_collections=[dnn_parent_scope])

            return logit, repr_emb

    drop_rate = 0.3
    activation_fn = tf.nn.relu
    with tf.variable_scope('logit'):

        input = cancel_feats
        input = tf.concat(input, axis=1)
        input = layers.batch_norm(input, is_training=is_training, activation_fn=None,
                                  variables_collections=[dnn_parent_scope])
        logit_cancel, repr_emb_cancel = auxiliary_net(input, activation_fn, drop_rate, is_training, reuse=False, scope='cancel')

        input = noroom_auxi_feats
        input = tf.concat(input, axis=1)
        input = layers.batch_norm(input, is_training=is_training, activation_fn=None,
                                  variables_collections=[dnn_parent_scope])
        logit_noroom_auxi, repr_emb_noroom = auxiliary_net(input, activation_fn, drop_rate, is_training, reuse=False, scope='noroom_auxi')

        input = feats
        input = tf.concat(input, axis=1)
        # input = tf.concat([input, repr_emb_noroom], axis=1)
        input = layers.batch_norm(input, is_training=is_training, activation_fn=None,
                                  variables_collections=[dnn_parent_scope])
        repr_emb = tf.concat([repr_emb_cancel, repr_emb_noroom], axis=1)
        logit_noroom = mlp_inference(input, repr_emb, activation_fn, drop_rate, is_training, reuse=False, scope='noroom')

    logit_dict = dict()
    logit_dict['logit_cancel'] = logit_cancel
    logit_dict['logit_noroom'] = logit_noroom
    logit_dict['logit_noroom_auxi'] = logit_noroom_auxi

    label_cancel = label_dict['pay']
    label_cancel = tf.cast(tf.equal(label_cancel, '1'), tf.float32)
    label_cancel = tf.reshape(label_cancel, [-1, 1])
    label_dict['label_cancel'] = label_cancel

    label_noroom = label_dict['click']
    label_noroom = tf.cast(tf.equal(label_noroom, '1'), tf.float32)
    label_noroom = tf.reshape(label_noroom, [-1, 1])
    label_dict['label_noroom'] = label_noroom

    label_noroom_all = outputs_dict['label_noromm_noord']
    label_noroom_all = tf.reshape(label_noroom_all, [-1, 1])
    label_dict['label_noroom_all'] = label_noroom_all

    is_noroom_sample = outputs_dict['is_noroom_sample']
    is_noroom_sample = tf.reshape(is_noroom_sample, [-1, 1])
    label_dict['is_noroom_sample'] = is_noroom_sample

    is_cancel_sample = outputs_dict['is_cancel_sample']
    is_cancel_sample = tf.reshape(is_cancel_sample, [-1, 1])
    label_dict['is_cancel_sample'] = is_cancel_sample

    return logit_dict, label_dict

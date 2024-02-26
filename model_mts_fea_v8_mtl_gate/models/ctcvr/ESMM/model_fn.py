# -*- coding: utf-8 -*-
import math
import tensorflow as tf
from tools import *

linear_parent_scope = "linear"
dnn_parent_scope = "dnn"


def add_tensor_summary(value, tag):
    tag = tag.replace(':', '_')
    tf.summary.scalar("%s/fraction_of_zero_values" % tag, tf.nn.zero_fraction(value))
    tf.summary.scalar("%s/mean" % tag, tf.reduce_mean(value))
    tf.summary.scalar("%s/max" % tag, tf.reduce_max(value))
    tf.summary.scalar("%s/min" % tag, tf.reduce_min(value))
    tf.summary.histogram("%s/activation" % tag, value)


def model_block(features, label_dict, fc_generator, is_training, keep_prob, global_step, params):
    # parse params
    black_list = params['black_list'] if 'black_list' in params else ""
    ########################################################
    # dnn
    tf.logging.info("building features...")
    outputs_dict = fc_generator.get_output_dict(features, black_list)

    seq_feat_list = ['shared_week', 'shared_days_diff', 'shared_holiday']

    id_feat_list = ['shid', 'hid', 'seller_id', 'star', 'brand', 'jk_type', 'domestic', 'province', 'city', 'district',
                    'geohash5', 'geohash6', 'is_credit_htl', 'business', 'types', 'hotel_types', 'hotel_tags',
                    'seller_type_desc', 'sub_seller_type_desc']
    idf_filter_names = ['shid', 'hid', 'geohash6', 'types', 'domestic', 'hotel_tags']

    tf.logging.info("finished build features:")
    for key in outputs_dict:
        tf.logging.info(key)
        tf.logging.info(outputs_dict[key])

    tf.logging.info("building features:")
    feats = []
    id_feats = []
    filter_names = id_feat_list + seq_feat_list + ['is_cancel_sample', 'is_noroom_sample',
                                                   'seller_3d_noroom_noord_rate', 'seller_28d_noroom_noord_rate']
    for key in outputs_dict:
        if key not in filter_names and 'seq_fea' not in key:
            tf.logging.info(key)
            tf.logging.info(outputs_dict[key])
            feats.append((key, outputs_dict[key]))
            tf.summary.scalar("%s/l2_norm" % key, tf.reduce_mean(tf.norm(outputs_dict[key], axis=1)))
        if key in id_feat_list and key not in idf_filter_names:
            id_feats.append((key, outputs_dict[key]))

    feats = [feat for _, feat in sorted(feats, key=lambda x: x[0])]
    id_feats = [feat for _, feat in sorted(id_feats, key=lambda x: x[0])]

    tf.logging.info("Total number of features: {}".format(len(feats)))

    for key in seq_feat_list:
        # if 'seq' in key:
        tf.logging.info("Seq features {}: {}".format(key, outputs_dict[key]))
        tf.summary.scalar("%s/l2_norm" % key, tf.reduce_mean(tf.norm(outputs_dict[key], axis=2)))

    # tf.summary.scalar("shared_week/l2_norm", tf.reduce_mean(tf.norm(outputs_dict["shared_week"], axis=2)))

    seq_feats = dict()
    # seq_feats["shared_week"] = tf.stack(outputs_dict["shared_week"], axis=1)  # (B,L,1)
    # seq_feats["shared_days_diff"] = tf.stack(outputs_dict["shared_days_diff"], axis=1)  # (B,L,1)
    # seq_feats["shared_holiday"] = tf.stack(outputs_dict["shared_holiday"], axis=1)  # (B,L,1)
    seq_feats["shid_seq_fea_ship_cnt"] = tf.stack(outputs_dict["shid_seq_fea_ship_cnt"], axis=1)  # (B,L,1)
    seq_feats["shid_seq_fea_refund_cnt"] = tf.stack(outputs_dict["shid_seq_fea_refund_cnt"], axis=1)  # (B,L,1)
    seq_feats["shid_seq_fea_no_room_cnt"] = tf.stack(outputs_dict["shid_seq_fea_no_room_cnt"], axis=1)  # (B,L,1)

    # seq_feats["hid_seq_fea_week"] = tf.stack(outputs_dict["hid_seq_fea_week"], axis=1)  # (B,L,1)
    # seq_feats["hid_seq_fea_days_diff"] = tf.stack(outputs_dict["hid_seq_fea_days_diff"], axis=1)  # (B,L,1)
    # seq_feats["hid_seq_fea_holiday"] = tf.stack(outputs_dict["hid_seq_fea_holiday"], axis=1)  # (B,L,1)
    seq_feats["hid_seq_fea_ship_cnt"] = tf.stack(outputs_dict["hid_seq_fea_ship_cnt"], axis=1)  # (B,L,1)
    seq_feats["hid_seq_fea_refund_cnt"] = tf.stack(outputs_dict["hid_seq_fea_refund_cnt"], axis=1)  # (B,L,1)
    seq_feats["hid_seq_fea_no_room_cnt"] = tf.stack(outputs_dict["hid_seq_fea_no_room_cnt"], axis=1)  # (B,L,1)
    seq_feats["hid_seq_fea_slr_op_cnt"] = tf.stack(outputs_dict["hid_seq_fea_slr_op_cnt"], axis=1)  # (B,L,1)

    # seq_feats["slr_seq_fea_week"] = tf.stack(outputs_dict["slr_seq_fea_week"], axis=1)  # (B,L,1)
    # seq_feats["slr_seq_fea_days_diff"] = tf.stack(outputs_dict["slr_seq_fea_days_diff"], axis=1)  # (B,L,1)
    # seq_feats["slr_seq_fea_holiday"] = tf.stack(outputs_dict["slr_seq_fea_holiday"], axis=1)  # (B,L,1)
    seq_feats["slr_seq_fea_ship_cnt"] = tf.stack(outputs_dict["slr_seq_fea_ship_cnt"], axis=1)  # (B,L,1)
    seq_feats["slr_seq_fea_refund_cnt"] = tf.stack(outputs_dict["slr_seq_fea_refund_cnt"], axis=1)  # (B,L,1)
    seq_feats["slr_seq_fea_no_room_cnt"] = tf.stack(outputs_dict["slr_seq_fea_no_room_cnt"], axis=1)  # (B,L,1)
    seq_feats["slr_seq_fea_slr_op_cnt"] = tf.stack(outputs_dict["slr_seq_fea_slr_op_cnt"], axis=1)  # (B,L,1)

    time_att_feats = [outputs_dict['diff'], outputs_dict['diff_hour'], outputs_dict['week_day']]

    activation_fn = tf.nn.relu

    def repr_net(inputs, activation, training, name):
        with tf.variable_scope(name):
            inputs = layers.batch_norm(inputs, is_training=training, activation_fn=None,
                                       variables_collections=[dnn_parent_scope])

            inputs = layers.fully_connected(inputs, 64, activation_fn=None, scope='ffn_1', weights_initializer=initializer,
                                            variables_collections=[dnn_parent_scope])

            inputs = layers.batch_norm(inputs, is_training=training, activation_fn=activation,
                                       variables_collections=[dnn_parent_scope])

            inputs = layers.fully_connected(inputs, 32, activation_fn=None, scope='ffn_2', weights_initializer=initializer,
                                            variables_collections=[dnn_parent_scope])

            inputs = layers.batch_norm(inputs, is_training=training, activation_fn=activation,
                                       variables_collections=[dnn_parent_scope])

            inputs = layers.fully_connected(inputs, 16, activation_fn=None, scope='ffn_3', weights_initializer=initializer,
                                            variables_collections=[dnn_parent_scope])
            inputs = layers.batch_norm(inputs, is_training=training, activation_fn=activation,
                                       variables_collections=[dnn_parent_scope])

            return inputs

    def mlp_pred(inputs, activation, drop_rate, training, name):
        with tf.variable_scope(name):
            inputs = layers.batch_norm(inputs, is_training=training, activation_fn=None,
                                       variables_collections=[dnn_parent_scope])

            inputs = layers.fully_connected(inputs, 256, activation_fn=None, scope='ffn_1', weights_initializer=initializer,
                                            variables_collections=[dnn_parent_scope])

            inputs = layers.batch_norm(inputs, is_training=training, activation_fn=activation,
                                       variables_collections=[dnn_parent_scope])

            inputs = layers.dropout(inputs, keep_prob=1.0 - drop_rate, is_training=training)

            inputs = layers.fully_connected(inputs, 128, activation_fn=None, scope='ffn_2', weights_initializer=initializer,
                                            variables_collections=[dnn_parent_scope])

            inputs = layers.batch_norm(inputs, is_training=training, activation_fn=activation,
                                       variables_collections=[dnn_parent_scope])

            inputs = layers.dropout(inputs, keep_prob=1.0 - drop_rate, is_training=training)

            inputs = layers.fully_connected(inputs, 64, activation_fn=None, scope='ffn_3', weights_initializer=initializer,
                                            variables_collections=[dnn_parent_scope])

            inputs = layers.batch_norm(inputs, is_training=training, activation_fn=activation,
                                       variables_collections=[dnn_parent_scope])

            logit = layers.fully_connected(inputs, 1, activation_fn=None, scope='logit',
                                           variables_collections=[dnn_parent_scope])

            return logit

    def seq_convolution(seq, in_channels, out_channels, kernel_size, training=True, name='conv'):
        # filter_conv = tf.get_variable("{}_filter".format(name),
        #                               shape=[kernel_size, in_channels, out_channels],
        #                               collections=[dnn_parent_scope, ops.GraphKeys.GLOBAL_VARIABLES,
        #                                            ops.GraphKeys.MODEL_VARIABLES])
        seq_len = seq.get_shape().as_list()[1]  # [B, L, 1]
        seq_bn = layers.batch_norm(seq, is_training=training, activation_fn=None,
                                   variables_collections=[dnn_parent_scope])
        seq_bn = tf.expand_dims(seq_bn, axis=3)  # [B, L, 1] --> [B, L, 1, 1]
        # convolution = tf.nn.conv1d(seq_bn, filters=filter_conv, stride=1, padding='SAME',
        #                            name='{}_conv'.format(name))  # (batch, in_width, out_channels)
        # value, ksize, strides, padding
        convolution = tf.nn.avg_pool(seq_bn, ksize=kernel_size, strides=[1, 1, 1, 1], padding='VALID',
                                     name='{}_conv'.format(name))  # [B, L', 1, 1]

        conv_len = convolution.get_shape().as_list()[1]  # L'
        convolution = tf.pad(convolution, [[0, 0], [seq_len - conv_len, 0], [0, 0], [0, 0]])  # [B, L, 1, 1]
        convolution = tf.squeeze(convolution, axis=3)  # [B, L, 1]
        # convolution = tf.nn.relu(convolution)
        # convolution = layers.batch_norm(convolution, is_training=training, activation_fn=tf.nn.relu,
        #                                 variables_collections=[dnn_parent_scope])
        return convolution

    drop_rate = 0.3
    initializer = tf.variance_scaling_initializer(scale=2.0, mode='fan_in', distribution='normal')
    with tf.variable_scope('logit'):
        # Convolution
        with tf.variable_scope('mts_extractor'):
            seq_names = seq_feats.keys()
            seq_names.sort()
            in_width = 28
            in_channel = 4
            filter_width = 14
            out_channel = 16
            filter_stride = 7
            output_width = in_width  # int(math.ceil(1.0 * (in_width - filter_width + 1) / filter_stride))

            seq_conv_list1 = list()
            seq_conv_list2 = list()
            seq_conv_list3 = list()
            seq_conv_list4 = list()
            for i in range(len(seq_names)):
                seq_name = seq_names[i]
                cur_seq_conv_list = list()
                cur_seq = seq_feats[seq_name]
                seq_conv_s1 = seq_convolution(cur_seq,
                                              in_channel,
                                              out_channel,
                                              kernel_size=[1, 3, 1, 1],
                                              training=is_training,
                                              name='{}_c1'.format(seq_name))  # [B, L, 1]

                seq_conv_s2 = seq_convolution(cur_seq,
                                              in_channel,
                                              out_channel,
                                              kernel_size=[1, 7, 1, 1],
                                              training=is_training,
                                              name='{}_c2'.format(seq_name))

                seq_conv_s3 = seq_convolution(cur_seq,
                                              in_channel,
                                              out_channel,
                                              kernel_size=[1, 14, 1, 1],
                                              training=is_training,
                                              name='{}_c3'.format(seq_name))  # (B, L', J), J = out_channels

                seq_conv_s4 = seq_convolution(cur_seq,
                                              in_channel,
                                              out_channel,
                                              kernel_size=[1, 28, 1, 1],
                                              training=is_training,
                                              name='{}_c4'.format(seq_name))

                seq_conv_list1.append(seq_conv_s1)  # (B, L', J), J = out_channels
                seq_conv_list2.append(seq_conv_s2)
                seq_conv_list3.append(seq_conv_s3)
                seq_conv_list4.append(seq_conv_s4)

                # conv_seqs = tf.stack(cur_seq_conv_list, axis=3)  # (B, L', J, P)
                #
                # var_att_input = layers.fully_connected(conv_seqs, 1, activation_fn=tf.nn.relu,
                #                                        scope='{}_a1'.format(seq_name),
                #                                        weights_initializer=initializer,
                #                                        variables_collections=[dnn_parent_scope])  # (B, L', J, 1)
                # var_att_input = tf.squeeze(var_att_input, axis=-1)  # (B, L', J)
                # var_att_logit = layers.fully_connected(var_att_input, len(cur_seq_conv_list), activation_fn=tf.nn.relu,
                #                                        scope='{}_a2'.format(seq_name),
                #                                        weights_initializer=initializer,
                #                                        variables_collections=[dnn_parent_scope])  # (B, L', P)
                # var_att_score = tf.nn.softmax(var_att_logit)  # (B, L', P)
                # var_att_score = tf.expand_dims(var_att_score, axis=3)  # (B, L', P, 1)
                # variable_attention = tf.matmul(conv_seqs, var_att_score)  # (B, L', J, 1)
                # variable_attention = tf.squeeze(variable_attention, axis=-1)  # (B, L', J)
                #
                # seq_conv_list.append(variable_attention)
            seq_comb1 = tf.concat(seq_conv_list1, axis=2)
            seq_repr1 = repr_net(seq_comb1, tf.nn.relu, is_training, 'seq_comb1')  # (B, L', J)
            seq_comb2 = tf.concat(seq_conv_list2, axis=2)
            seq_repr2 = repr_net(seq_comb2, tf.nn.relu, is_training, 'seq_comb2')  # (B, L', J)
            seq_comb3 = tf.concat(seq_conv_list3, axis=2)
            seq_repr3 = repr_net(seq_comb3, tf.nn.relu, is_training, 'seq_comb3')  # (B, L', J)
            seq_comb4 = tf.concat(seq_conv_list4, axis=2)
            seq_repr4 = repr_net(seq_comb4, tf.nn.relu, is_training, 'seq_comb4')  # (B, L', J)

            conv_seqs = tf.stack([seq_repr1, seq_repr2, seq_repr3, seq_repr4], axis=3)  # (B, L', J, N*P)
            var_att_input = layers.fully_connected(conv_seqs, 1, activation_fn=tf.nn.relu, scope='v_att1',
                                                   weights_initializer=initializer,
                                                   variables_collections=[dnn_parent_scope])  # (B, L', J, 1)
            var_att_input = tf.squeeze(var_att_input, axis=-1)  # (B, L', J)
            var_att_logit = layers.fully_connected(var_att_input, 4, activation_fn=tf.nn.relu,
                                                   scope='v_att2',
                                                   weights_initializer=initializer,
                                                   variables_collections=[dnn_parent_scope])  # (B, L', N*P)
            var_att_score = tf.nn.softmax(var_att_logit)  # (B, L', N*P)
            var_att_score = tf.expand_dims(var_att_score, axis=3)  # (B, L', N*P, 1)
            variable_attention = tf.matmul(conv_seqs, var_att_score)  # (B, L', J, 1)
            variable_attention = tf.squeeze(variable_attention, axis=-1)  # (B, L', J)

            time_att_input_raw = tf.transpose(variable_attention, perm=[0, 2, 1])  # (B, J, L')
            # time_att_input = layers.fully_connected(time_att_input_raw, 1, activation_fn=tf.nn.relu, scope='t_att1',
            #                                         weights_initializer=initializer,
            #                                         variables_collections=[dnn_parent_scope])  # (B, J, 1)
            # time_att_input = tf.squeeze(time_att_input, axis=-1)  # (B, J)

            time_att_feats = tf.concat(time_att_feats, axis=1)
            time_att_feats = layers.batch_norm(time_att_feats, is_training=is_training, activation_fn=None,
                                               variables_collections=[dnn_parent_scope])
            time_att_feats = layers.fully_connected(time_att_feats, 4, activation_fn=tf.nn.relu, scope='ct_att1',
                                                    weights_initializer=initializer,
                                                    variables_collections=[dnn_parent_scope])

            time_att_input = time_att_feats  # tf.concat(time_att_feats + [time_att_input], axis=1)
            time_att_logit = layers.fully_connected(time_att_input, output_width, activation_fn=tf.nn.relu, scope='t_att2',
                                                    weights_initializer=initializer,
                                                    variables_collections=[dnn_parent_scope])  # (B, L')
            time_att_score = tf.nn.softmax(time_att_logit)  # (B, L')
            time_att_score = tf.expand_dims(time_att_score, axis=2)  # (B, L', 1)
            time_attention = tf.matmul(time_att_input_raw, time_att_score)  # (B, J, 1)
            mtl_att_vector = tf.squeeze(time_attention, axis=-1)  # (B, J)

            add_tensor_summary(mtl_att_vector, 'mtl_att_vector')
            tf.summary.scalar("mtl_att_vector/l2_norm", tf.reduce_mean(tf.norm(mtl_att_vector, axis=1)))

        input = id_feats
        input = tf.concat(input, axis=1)
        cancel_repr = repr_net(input, activation_fn, is_training, 'cancel_repr')

        # MLP
        # cancel_repr_s = tf.stop_gradient(cancel_repr)
        # cancel_repr_s = tf.cond(global_step < 6000, lambda: tf.zeros_like(cancel_repr_s), lambda: cancel_repr_s)
        input = feats + [mtl_att_vector, cancel_repr]
        input = tf.concat(input, axis=1)

        # input = layers.batch_norm(input, is_training=is_training, activation_fn=None,
        #                           variables_collections=[dnn_parent_scope])
        #
        # input = layers.fully_connected(input, 256, activation_fn=None, scope='ffn_1', weights_initializer=initializer,
        #                                variables_collections=[dnn_parent_scope])
        #
        # input = layers.batch_norm(input, is_training=is_training, activation_fn=activation_fn,
        #                           variables_collections=[dnn_parent_scope])
        #
        # input = layers.dropout(input, keep_prob=1.0 - drop_rate, is_training=is_training)
        #
        # input = layers.fully_connected(input, 128, activation_fn=None, scope='ffn_2', weights_initializer=initializer,
        #                                variables_collections=[dnn_parent_scope])
        #
        # input = layers.batch_norm(input, is_training=is_training, activation_fn=activation_fn,
        #                           variables_collections=[dnn_parent_scope])
        #
        # input = layers.dropout(input, keep_prob=1.0 - drop_rate, is_training=is_training)
        #
        # input = layers.fully_connected(input, 64, activation_fn=None, scope='ffn_3', weights_initializer=initializer,
        #                                variables_collections=[dnn_parent_scope])
        #
        # input = layers.batch_norm(input, is_training=is_training, activation_fn=activation_fn,
        #                           variables_collections=[dnn_parent_scope])
        #
        # logit = layers.fully_connected(input, 1, activation_fn=None, scope='logit',
        #                                variables_collections=[dnn_parent_scope])

        logit_noroom = mlp_pred(inputs=input, activation=activation_fn, drop_rate=drop_rate, training=is_training, name='nr_net')

        # cancel_repr_s = tf.stop_gradient(cancel_repr)
        # cancel_repr = tf.cond(global_step < 6000, lambda: cancel_repr, lambda: cancel_repr_s)
        logit_cancel = layers.fully_connected(cancel_repr, 1, activation_fn=None, scope='cancel_logit',
                                              variables_collections=[dnn_parent_scope])
        # logit_cancel = mlp_pred(inputs=cancel_repr, activation=activation_fn, drop_rate=drop_rate, training=is_training, name='cl_net')

    logit_dict = {}
    logit_dict['logit'] = logit_noroom
    logit_dict['logit_noroom'] = logit_noroom
    logit_dict['logit_cancel'] = logit_cancel

    logit_dict['is_cancel_sample'] = outputs_dict['is_cancel_sample']
    logit_dict['is_noroom_sample'] = outputs_dict['is_noroom_sample']

    logit_dict['var_att_score'] = tf.squeeze(var_att_score, axis=-1)  # (B, L'*P)
    logit_dict['time_att_score'] = tf.squeeze(time_att_score, axis=-1)  # (B, L')

    label_click = label_dict['click']
    label_click = tf.cast(tf.equal(label_click, '1'), tf.float32)
    label_click = tf.reshape(label_click, [-1, 1])
    label_dict['click'] = label_click

    label_pay = label_dict['pay']
    label_pay = tf.cast(tf.equal(label_pay, '1'), tf.float32)
    label_pay = tf.reshape(label_pay, [-1, 1])
    label_dict['pay'] = label_pay

    return logit_dict, label_dict

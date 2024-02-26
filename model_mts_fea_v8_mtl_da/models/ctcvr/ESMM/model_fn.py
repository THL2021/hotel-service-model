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
    # idf_filter_names = ['shid', 'hid', 'geohash6', 'types', 'domestic', 'hotel_tags']
    idf_filter_names = ['hid']

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

            all_seq_feature = [seq_feats[name] for name in seq_names]
            seq_features = tf.concat(all_seq_feature, axis=2)

            # tf.nn.rnn_cell.BasicLSTMCell
            # LSTM
            # class LSTMStateTuple(_LSTMStateTuple):
            # """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.
            #
            # Stores two elements: `(c, h)`, in that order. Where `c` is the hidden state
            # and `h` is the output.
            #
            # Only used when `state_is_tuple=True`.
            # """
            seq_features = layers.batch_norm(seq_features, is_training=is_training, activation_fn=None,
                                             scope='bn1',
                                             variables_collections=[dnn_parent_scope])
            HIDDEN_SIZE = 16
            SEQ_LEN = 28
            lstm = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True)
            batch_size = seq_features.get_shape().as_list()[0]

            att_series_logits = []
            for k in range(len(all_seq_feature)):
                seq_input = seq_features[:, :, k]  # B, L
                att_logit = layers.fully_connected(seq_input, SEQ_LEN, activation_fn=tf.nn.tanh,
                                                   scope='s_att',
                                                   weights_initializer=initializer,
                                                   variables_collections=[dnn_parent_scope],
                                                   reuse=tf.AUTO_REUSE)  # B, L
                att_series_logits.append(att_logit)

            c_list = []
            h_list = []
            new_seq_step = []
            init_sv = tf.tile(tf.expand_dims(seq_features[:, 1, 1], axis=1), [1, HIDDEN_SIZE])
            c = h = init_sv
            with tf.variable_scope('lstm1'):
                for t in range(SEQ_LEN):
                    if t > 0:
                        tf.get_variable_scope().reuse_variables()
                    c_list.append(c)
                    h_list.append(h)
                    att_input = tf.concat([c, h], axis=1)  # B, 2m
                    att_input = layers.fully_connected(att_input, SEQ_LEN, activation_fn=None, scope='att1',
                                                       weights_initializer=initializer,
                                                       variables_collections=[dnn_parent_scope],
                                                       reuse=tf.AUTO_REUSE)  # B, L
                    k_logits = []
                    for att in att_series_logits:
                        att_input = tf.nn.tanh(att_input + att)  # B, L
                        att_logit = layers.fully_connected(att_input, 1, activation_fn=None, scope='att2',
                                                           weights_initializer=initializer,
                                                           variables_collections=[dnn_parent_scope],
                                                           reuse=tf.AUTO_REUSE)  # B, 1
                        k_logits.append(att_logit)

                    k_var_logits = tf.concat(k_logits, axis=1)  # B, N
                    k_var_score = tf.nn.softmax(k_var_logits)  # B, N

                    step_t = seq_features[:, t, :]  # B, N
                    new_step_t = k_var_score * step_t

                    new_seq_step.append(new_step_t)

                    output, (c, h) = lstm(new_step_t, (c, h))

            # LSTM
            with tf.variable_scope('lstm2'):
                HIDDEN_SIZE = 16
                SEQ_LEN = 28
                rnn_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True)
                print("new_seq_step:", new_seq_step)
                seq_features = tf.stack(new_seq_step, axis=1)
                seq_features = layers.batch_norm(seq_features, is_training=is_training, activation_fn=None,
                                                 scope='bn2',
                                                 variables_collections=[dnn_parent_scope])
                print("seq_features:", seq_features)
                seq_length = tf.tile(tf.constant([SEQ_LEN], dtype=tf.int32), [tf.shape(seq_features)[0]])
                outputs, (c, h) = tf.nn.dynamic_rnn(rnn_cell, seq_features, sequence_length=seq_length, dtype=tf.float32)

            mtl_att_vector = h

            add_tensor_summary(mtl_att_vector, 'mtl_att_vector')
            tf.summary.scalar("mtl_att_vector/l2_norm", tf.reduce_mean(tf.norm(mtl_att_vector, axis=1)))

        input = id_feats
        input = tf.concat(input, axis=1)
        cancel_repr = repr_net(input, activation_fn, is_training, 'cancel_repr')

        # MLP
        # cancel_repr_s = tf.stop_gradient(cancel_repr)
        # cancel_repr_s = tf.cond(global_step < 6000, lambda: tf.zeros_like(cancel_repr_s), lambda: cancel_repr_s)
        # input = feats + [mtl_att_vector, cancel_repr]
        input = feats + [mtl_att_vector]
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

    label_click = label_dict['click']
    label_click = tf.cast(tf.equal(label_click, '1'), tf.float32)
    label_click = tf.reshape(label_click, [-1, 1])
    label_dict['click'] = label_click

    label_pay = label_dict['pay']
    label_pay = tf.cast(tf.equal(label_pay, '1'), tf.float32)
    label_pay = tf.reshape(label_pay, [-1, 1])
    label_dict['pay'] = label_pay

    return logit_dict, label_dict

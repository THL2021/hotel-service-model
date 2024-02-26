# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import optimizers
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops

linear_parent_scope = "linear"
dnn_parent_scope = "dnn"


def make_training_op(training_loss, global_step, is_sequence_train):
    _DNN_LEARNING_RATE = 0.001
    _LINEAR_LEARNING_RATE = 0.005
    _GRADIENT_CLIP_NORM = 100.0

    warm_up_learning_rate = 0.0001
    warm_up_step = 1000
    init_learning_rate = 0.001
    decay_steps = 2000
    decay_rate = 0.7
    learning_rate = tf.train.smooth_exponential_decay(warm_up_learning_rate,
                                                      warm_up_step,
                                                      init_learning_rate,
                                                      global_step,
                                                      decay_steps,
                                                      decay_rate)

    with ops.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_ops = []
        # linear_optimizer = tf.train.FtrlOptimizer(
        #     learning_rate=_LINEAR_LEARNING_RATE,
        #     learning_rate_power=-0.5,
        #     initial_accumulator_value=1.0,
        #     l1_regularization_strength=0.1,
        #     l2_regularization_strength=0.01
        # )
        if is_sequence_train:
            dnn_optimizer = tf.train.AdamOptimizer(_DNN_LEARNING_RATE)
        else:
            dnn_optimizer = tf.train.AdamOptimizer(learning_rate)
        train_ops.append(
            optimizers.optimize_loss(
                loss=training_loss,
                global_step=global_step,
                learning_rate=_DNN_LEARNING_RATE,
                optimizer=dnn_optimizer,
                variables=ops.get_collection(dnn_parent_scope),
                name=dnn_parent_scope,
                clip_gradients=None,
                increment_global_step=None))
        tf.logging.info(
            "optimizer scope {} variables: {}".format(dnn_parent_scope, ops.get_collection(dnn_parent_scope)))
        train_op = control_flow_ops.group(*train_ops)
        with ops.control_dependencies([train_op]):
            with ops.colocate_with(global_step):
                return state_ops.assign_add(global_step, 1).op


def sequence_train_op_gen(patterns, learning_rate, global_step):
    dependency = None
    # train_ops = []
    for pattern in patterns:
        if dependency is None:
            tf.logging.info(pattern[2])
            dependency = train_op_gen(pattern[0], global_step, learning_rate, pattern[1], pattern[2])
        else:
            tf.logging.info(pattern[2])
            with tf.control_dependencies([dependency]):
                dependency = train_op_gen(pattern[0], global_step, learning_rate, pattern[1], pattern[2])
        # train_ops.append(dependency)
    train_op = dependency
    # train_op = control_flow_ops.group(*train_ops)
    with ops.control_dependencies([train_op]):
        with ops.colocate_with(global_step):
            return state_ops.assign_add(global_step, 1).op


def make_training_op_transfer_WDGD(training_loss, global_step, is_sequence_train):
    class_target_loss = training_loss['class_target_loss']
    class_source_loss = training_loss['class_source_loss']

    common_spe_wd_loss = training_loss["common_spe_wd_loss"]
    common_spe_gradient_penalty = training_loss["common_spe_gradient_penalty"]

    adversial_w = 1.0
    critic_train_num = 5
    wd_w = 0.001
    patterns = []

    patterns.append(
        (class_target_loss + class_source_loss + wd_w * common_spe_wd_loss,
         ["embedding_layer", "generator_target_spe", "generator_source_spe", "generator_common", "target_classifer",
          "source_classifer"], "step_G"))

    for i in range(critic_train_num):
        patterns.append((adversial_w * common_spe_gradient_penalty - common_spe_wd_loss,
                         ["critic"], "step_D_{}".format(i + 1)))

    train_op = sequence_train_op_gen(patterns, 0.0001, global_step)
    return train_op


def make_training_op_transfer_MULTI_TASK(training_loss, global_step, is_sequence_train):
    class_target_loss = training_loss['class_target_loss']
    class_source_loss = training_loss['class_source_loss']

    patterns = []  # (loss, var_scopes, name)
    patterns.append((class_target_loss + class_source_loss,
                     ["embedding_layer", "generator_common", "target_classifer", "source_classifer"], "A"))
    train_op = sequence_train_op_gen(patterns, 0.001, global_step)
    return train_op


def make_training_op_transfer_TADA(training_loss, global_step, is_sequence_train):
    class_target_loss = training_loss['class_target_loss']
    class_source_loss = training_loss['class_source_loss']

    dis_loss_target = training_loss['dis_loss_target']
    dis_loss_source = training_loss['dis_loss_source']
    adv_loss = training_loss['adv_loss']

    critic_train_num = 5
    adversial_w = 0.001
    patterns = []

    patterns.append(
        (
            class_target_loss + class_source_loss + adversial_w * dis_loss_target + adversial_w * dis_loss_source - adversial_w * adv_loss,
            ["embedding_layer", "generator_target_spe", "generator_source_spe", "generator_common", "target_classifer",
             "source_classifer"], "step_G"))

    for i in range(critic_train_num):
        patterns.append((adv_loss, ["discriminator"],
                         "step_D_{}".format(i + 1)))

    train_op = sequence_train_op_gen(patterns, 0.001, global_step)
    return train_op


def train_op_gen(loss, global_step, learning_rate, var_scopes, name):
    clip_gradients = None
    dnn_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="Adam_{}".format(name))
    vars_all = tf.trainable_variables()
    vars = None
    if var_scopes is not None:
        vars = []
        for scope in var_scopes:
            vars_tmp = [var for var in vars_all if scope in var.name]
            vars.extend(vars_tmp)
            
    if not vars:
        # default for all
        vars = None

    update_vars = None
    if var_scopes is not None:
        # update_vars = [], nonthing to update
        # update_vars = None, default
        update_vars = []
        for scope in var_scopes:
            update_vars_tmp = tf.get_collection("update_ops_{}".format(scope))
            update_vars.extend(update_vars_tmp)

    train_op = optimizers.optimize_loss(
        loss=loss,
        global_step=global_step,
        learning_rate=None,
        optimizer=dnn_optimizer,
        update_ops=update_vars,
        variables=vars,
        name=name,
        clip_gradients=clip_gradients,
        increment_global_step=None)
    tf.logging.info(
        "optimizer name {} variables: {}".format(name, vars))
    tf.logging.info(
        "optimizer name {} update_vars: {}".format(name, update_vars))
    return train_op

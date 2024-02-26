"""for special loss function"""
import tensorflow as tf


def binary_cross_entropy_with_auc_loss(y_true, y_pred):
    logloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))
    y_pred = tf.sigmoid(y_pred)
    eps = 1e-6
    y_pred_clipped = tf.clip_by_value(y_pred, eps, 1 - eps)
    y_pred_score = tf.log(y_pred_clipped / (1.0 - y_pred_clipped))

    y_pred_zero_max = tf.reduce_max(y_pred_score * tf.cast(y_true < 1, tf.float32))
    rank_loss = y_pred_score - y_pred_zero_max
    rank_loss = rank_loss * y_true
    rank_loss = tf.square(tf.clip_by_value(rank_loss, -100, 0))
    rank_loss1 = tf.reduce_sum(rank_loss) / (tf.reduce_sum(tf.cast(y_true > 0, tf.float32)) + 1)

    y_pred_one_min = tf.reduce_min(y_pred_score * tf.cast(y_true > 0, tf.float32))
    rank_loss = y_pred_score - y_pred_one_min
    rank_loss = rank_loss * tf.cast(y_true < 1, tf.float32)
    rank_loss = tf.square(tf.clip_by_value(rank_loss, 0, 100))
    rank_loss2 = tf.reduce_sum(rank_loss) / (tf.reduce_sum(tf.cast(y_true < 1, tf.float32)) + 1)

    y_pred_zero_min = tf.reduce_min(y_pred_score * tf.cast(y_true < 1, tf.float32))
    rank_loss = y_pred_score - y_pred_zero_min
    rank_loss = rank_loss * y_true
    rank_loss = tf.square(tf.clip_by_value(rank_loss, -100, 0))
    rank_loss3 = tf.reduce_sum(rank_loss) / (tf.reduce_sum(tf.cast(y_true > 0, tf.float32)) + 1)

    y_pred_one_max = tf.reduce_max(y_pred_score * tf.cast(y_true > 0, tf.float32))
    rank_loss = y_pred_score - y_pred_one_max
    rank_loss = rank_loss * tf.cast(y_true < 1, tf.float32)
    rank_loss = tf.square(tf.clip_by_value(rank_loss, 0, 100))
    rank_loss4 = tf.reduce_sum(rank_loss) / (tf.reduce_sum(tf.cast(y_true < 1, tf.float32)) + 1)

    return (rank_loss1 + rank_loss2 + rank_loss3 + rank_loss4 + 1) * logloss

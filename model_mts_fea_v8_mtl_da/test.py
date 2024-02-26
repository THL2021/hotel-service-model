import random
import tensorflow as tf
import numpy as np

from tensorflow.contrib.keras.api.keras.layers import LSTM

# print np.random.rand(3, 7, 2)
seq_features = tf.constant(np.random.rand(3, 7, 2), dtype=tf.float32)
HIDDEN_SIZE = 16
# SEQ_LEN = 28
# lstm1, state_h, state_c = LSTM(HIDDEN_SIZE, return_sequences=True, return_state=False)(seq_features)
# print type(state_h),lstm1.shape,state_h.shape,state_c.shape

# seq_features = layers.batch_norm(seq_features, is_training=is_training, activation_fn=None,
#                                  variables_collections=[dnn_parent_scope])
HIDDEN_SIZE = 16
SEQ_LEN = 7
lstm = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True)
batch_size = seq_features.get_shape().as_list()[0]

c_list = []
h_list = []
c, h = (tf.zeros(shape=(batch_size, HIDDEN_SIZE)), tf.zeros(shape=(batch_size, HIDDEN_SIZE)))
for t in range(SEQ_LEN):
    print seq_features[:, t, :]
    print seq_features[:, :, 1]
    output, (c, h) = lstm(seq_features[:, t, :], (c, h))
    c_list.append(c)
    h_list.append(h)

print c_list
print h_list

import bisect


def inversions_bisect(l):
    ri, res = [], 0
    for i in reversed(range(0, len(l))):
        bs = bisect.bisect_left(ri, l[i])
        res += bs
        ri.insert(bs, l[i])
    return res

l = 'x,x,x'.split(',')
print l
lx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
print 'inversions_bisect:',inversions_bisect([1,3,2])
print 'inversions_bisect:',inversions_bisect([11, 6, 5, 11, 6, 5, 11, 6, 5, 11, 6, 5, 11, 6, 5,11, 6, 5, 11, 6, 5, 11, 6, 5, 11, 6, 5, 11, 6, 5,11, 6, 5, 11, 6, 5, 11, 6, 5, 11, 6, 5, 11, 6, 5,11, 6, 5, 11, 6, 5, 11, 6, 5, 11, 6, 5, 11, 6, 5])

print 'range',range(10)[::-1]


lxx = [1,2,3,4,5,6]
print lxx[-2:]

w1 = tf.Variable(100, name="w", trainable=False)
w2 = tf.to_float(w1) / 2.5  # tf.div(tf.to_float(w1), 2.5)
w3 = w1.assign_add(1)
#
# a = [1, 2, 3, 4, 5]
# print a[:-2], a[-2:]
# print [[' ']] * 5
#
#
# ls = ['odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210618', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210619', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210620', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210621', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210622', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210623', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210624', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210625', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210626', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210627', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210628', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210629', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210630', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210701', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210702', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210703', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210704', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210705', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210706', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210707', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210708', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210709', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210710', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210711', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210712', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210713', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210714', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210715', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210716', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210717', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210718', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210719', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210720', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210721', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210722', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210723', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210724', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210725', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210726', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210727', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210728', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210729', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210730', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210731', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210801', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210802', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210803', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210804', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210805', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210806', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210807', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210808']
# ls = list(ls)
# print random.shuffle(ls)
# print "Shuffed files: ", ls
print 3289 % 100
f1 = tf.constant([[1.0, 2, 3],
                  [3, 2, 6]])
f2 = tf.constant([[2.0, 1, 5],
                  [5, 1, 3]])
f = [f1, f2]

a = tf.random_uniform([1, len(f), 1], minval=0, maxval=1)
a = tf.where(a > 0.5, tf.ones_like(a), tf.zeros_like(a))
b = a
c = tf.stack(f, axis=1) * a
d = tf.unstack(c, axis=1)

input = tf.concat(d, axis=1)
input2 = tf.concat(f, axis=1)

b1 = tf.constant(False, dtype=tf.bool)

input = tf.cond(b1, lambda: input, lambda: input2)
# print b1

t1s = tf.constant([[[1], [2]],
                   [[2], [3]],
                   [[3], [1]]])

t2s = tf.squeeze(t1s, axis=-1)

print 'shape:',t1s.shape, t2s.shape
# b = tf.split(a, num_or_size_splits=3, axis=1)
# # c = b * b
# print b
# print 'shape:',a.shape, a.shape[1]
inti = tf.global_variables_initializer()
with tf.Session() as sess:
    # for i in range(1):
    sess.run(inti)
    a_v, b_v = sess.run([w2, w3])
    print 'a_v:', b_v, '\n'
    a_v, b_v = sess.run([w2, w3])
    print 'a_v:', b_v, '\n'
    a_v, b_v = sess.run([w2, w3])
    print 'a_v:', b_v, '\n'
    # a_v, b_v, c_v = sess.run([inti, w2, input])
    # print 'a_v:', b_v, '\n'
    # print i, '\nb_v:', b_v, '\n'
    # print i, '\nc_v:', c_v, '\n'
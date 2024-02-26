# import random
#
# a = [1, 2, 3, 4, 5]
# print a[:-2], a[-2:]
# print [[' ']] * 5
#
#
# ls = ['odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210618', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210619', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210620', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210621', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210622', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210623', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210624', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210625', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210626', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210627', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210628', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210629', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210630', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210701', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210702', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210703', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210704', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210705', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210706', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210707', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210708', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210709', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210710', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210711', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210712', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210713', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210714', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210715', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210716', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210717', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210718', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210719', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210720', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210721', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210722', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210723', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210724', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210725', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210726', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210727', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210728', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210729', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210730', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210731', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210801', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210802', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210803', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210804', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210805', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210806', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210807', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210808']
# ls = list(ls)
# print "Shuffed files: ", ls
# print random.shuffle(ls)


import tensorflow as tf
a = tf.constant([-1, 0, 1])
with tf.Session() as sess:
    # sess.watch(a)
    y = tf.nn.relu(a)
    grads = tf.gradients(y, a)
    a, y, grads = sess.run([a, y, grads])
print('x:', a)
print('y:', y)
print('grad:', grads)


class Node:
    def __init__(self):
        self.v = 0.0
        self.l = None
        self.r = None

    def add(self, v):
        if v == 0:
            return v
        return v + self.add(v - 1)

node = Node()
node.v = 10

print(node.add(3))
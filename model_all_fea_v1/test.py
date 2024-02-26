import random

a = [1, 2, 3, 4, 5]
# print a[:-2], a[-2:]
# print [[' ']] * 5


ls = ['odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210618', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210619', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210620', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210621', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210622', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210623', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210624', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210625', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210626', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210627', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210628', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210629', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210630', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210701', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210702', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210703', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210704', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210705', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210706', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210707', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210708', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210709', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210710', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210711', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210712', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210713', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210714', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210715', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210716', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210717', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210718', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210719', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210720', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210721', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210722', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210723', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210724', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210725', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210726', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210727', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210728', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210729', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210730', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210731', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210801', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210802', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210803', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210804', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210805', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210806', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210807', 'odps://trip_algo_dev/tables/hotel_order_no_room_train_data_fg_v1/ds=20210808']
ls = list(ls)
# print random.shuffle(ls)
# print "Shuffed files: ", ls
model_version = ('new_pssmart_1','new_msf_v1_1','new_nrm_mtl_seq_v1_1','new_nrm_mtl_v1_1','new_maf_v1_1', 'new_nrm_mtl_seq_v2_1')

print 'new_pssm2art_1' in model_version

s = set()
s.add(1)
s.add(3)
s.add(15)
print s
print 3 not in s

l = [1, 2, 3, 4, 5]
print l[:3]
print l[::2]
print l[::-1]
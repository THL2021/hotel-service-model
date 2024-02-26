import tensorflow as tf
import rtp_fg
import json
import random
from fg_worker import FeatureColumnGenerator

table_col_num = 5
fg_file_path = './hotel_no_room_dnn_feat_config.json'
fg_tf_file_path = './hotel_no_room_dnn_feat_config_tf.json'


def parse_fg_transfer(data_batch, FLAGS):
    tf.logging.info("loading json config...")
    with open(fg_file_path, 'r') as f:
        feature_configs_java = json.load(f)
    with open(fg_tf_file_path, 'r') as f:
        feature_configs_tf = json.load(f)

    train_iterator_target = data_batch["train_iterator_target"]
    train_iterator_source = data_batch["train_iterator_source"]

    tf.logging.info("java fg parsing...")
    features = {}
    label_dict = {}
    features_target, label_dict_target = parser(train_iterator_target, feature_configs_java)
    features_source, label_dict_source = parser(train_iterator_source, feature_configs_java)
    features["features_target"] = features_target
    features["features_source"] = features_source
    label_dict["label_dict_target"] = label_dict_target
    label_dict["label_dict_source"] = label_dict_source

    tf.logging.info("finished java fg parsing...")
    tf.logging.info("tf fg...")
    fc_generator = FeatureColumnGenerator(feature_configs_tf)
    tf.logging.info("finished tf fg...")
    ########################################################
    params = {name: value for name, value in FLAGS.__flags.items()}
    print("print params...")
    for key in params:
        print("params {}: {}".format(key, params[key]))
    return features, label_dict, fc_generator, params


duplication_fea_list = [('shid', 'shid_dp1'), ('hid', 'hid_dp1'), ('seller_id', 'seller_id_dp1'), ('star', 'star_dp1'),
                        ('brand', 'brand_dp1'), ('jk_type', 'jk_type_dp1'), ('domestic', 'domestic_dp1'),
                        ('province', 'province_dp1'), ('city', 'city_dp1'), ('district', 'district_dp1'),
                        ('geohash5', 'geohash5_dp1'), ('geohash6', 'geohash6_dp1'),
                        ('is_credit_htl', 'is_credit_htl_dp1'), ('business', 'business_dp1'),
                        ('types', 'types_dp1'), ('hotel_types', 'hotel_types_dp1'), ('hotel_tags', 'hotel_tags_dp1'),
                        ('seller_type_desc', 'seller_type_desc_dp1'),
                        ('sub_seller_type_desc', 'sub_seller_type_desc_dp1'),

                        ('shid', 'shid_dp2'), ('hid', 'hid_dp2'), ('seller_id', 'seller_id_dp2'), ('star', 'star_dp2'),
                        ('brand', 'brand_dp2'), ('jk_type', 'jk_type_dp2'), ('domestic', 'domestic_dp2'),
                        ('province', 'province_dp2'), ('city', 'city_dp2'), ('district', 'district_dp2'),
                        ('geohash5', 'geohash5_dp2'), ('geohash6', 'geohash6_dp2'),
                        ('is_credit_htl', 'is_credit_htl_dp2'), ('business', 'business_dp2'),
                        ('types', 'types_dp2'), ('hotel_types', 'hotel_types_dp2'), ('hotel_tags', 'hotel_tags_dp2'),
                        ('seller_type_desc', 'seller_type_desc_dp2'),
                        ('sub_seller_type_desc', 'sub_seller_type_desc_dp2')]


def parse_fg(data_batch, FLAGS):
    tf.logging.info("loading json config...")
    with open(fg_file_path, 'r') as f:
        feature_configs_java = json.load(f)
    with open(fg_tf_file_path, 'r') as f:
        feature_configs_tf = json.load(f)

    tf.logging.info("java fg parsing...")
    features, label_dict = parser(data_batch, feature_configs_java)

    for raw_name, new_name in duplication_fea_list:
        if raw_name in features:
            features[new_name] = features[raw_name]

    tf.logging.info("finished java fg parsing...")
    tf.logging.info("tf fg...")
    fc_generator = FeatureColumnGenerator(feature_configs_tf)
    tf.logging.info("finished tf fg...")
    ########################################################
    params = {name: value for name, value in FLAGS.__flags.items()}
    print("print params...")
    for key in params:
        print("params {}: {}".format(key, params[key]))
    return features, label_dict, fc_generator, params


def parser(batch, feature_configs):
    columns = batch.get_next()
    key, feature, label, label_pay, _ = columns
    # shape must be rank 1
    feature = tf.reshape(feature, [-1, 1])
    feature = tf.squeeze(feature, axis=1)
    features = rtp_fg.parse_genreated_fg(feature_configs, feature)

    label_dict = {}
    label_dict['click'] = label
    label_dict['pay'] = label_pay
    return features, label_dict


def input_fn(files, batch_size, mode, is_sequence_train, slice_id, slice_count, epochs=1):
    tf.logging.info("slice_count:{}, slice_id:{}".format(slice_count, slice_id))
    if mode == 'train':
        if is_sequence_train:
            # sequence train
            dataset = tf.data.TableRecordDataset(files, [[' ']] * table_col_num, slice_id=slice_id,
                                                 slice_count=slice_count).repeat(epochs).batch(batch_size)

            # dataset = tf.data.TableRecordDataset(files, [[' ']] * table_col_num, slice_id=slice_id,
            #                                      slice_count=slice_count).batch(batch_size).filter(
            #     lambda x1, x2, x3, x4, x5: tf.equal(tf.shape(x1)[0], batch_size))

        else:
            # global train
            files = list(files)
            random.shuffle(files)
            print "Shuffed files: ", files
            dataset = tf.data.TableRecordDataset(files, [[' ']] * table_col_num, slice_id=slice_id,
                                                 slice_count=slice_count).shuffle(
                buffer_size=200 * batch_size).repeat(epochs).batch(batch_size)
    elif mode == 'dev':
        dataset = tf.data.TableRecordDataset(files, [[' ']] * table_col_num, slice_id=slice_id,
                                             slice_count=slice_count).repeat().batch(batch_size)
    return dataset


def input_fn_normal(files, batch_size, mode):
    if mode == 'train':
        dataset = tf.data.TableRecordDataset(files, [[' ']] * table_col_num).shuffle(
            buffer_size=200 * batch_size).repeat().batch(batch_size)
    elif mode == 'dev':
        dataset = tf.data.TableRecordDataset(files, [[' ']] * table_col_num).batch(batch_size)
    elif mode == 'infer':
        dataset = tf.data.TableRecordDataset(files, [[' ']] * table_col_num).batch(batch_size)
    return dataset


def transform_fg(dic_0):
    remove_list = ["is_wide", "value_type", "hash_bucket_size", "embed_size", "shared", "shared_matrix_name",
                   "normalizer", "is_list"]
    for dic_1 in dic_0['features']:
        if "sequence_name" in dic_1:
            for dic_2 in dic_1['features']:
                for remove_key in remove_list:
                    if remove_key in dic_2:
                        del dic_2[remove_key]
        else:
            for remove_key in remove_list:
                if remove_key in dic_1:
                    del dic_1[remove_key]
    return dic_0


def clear_dfs_dir(model_dir):
    from tensorflow.python.lib.io import file_io
    from tensorflow.python.platform import tf_logging as logging
    try:
        logging.info('clear releasing room if exists')
        file_exist = file_io.is_directory(model_dir)
        if file_exist:
            file_io.delete_file(model_dir)
            logging.info("model releasing success, good luck!")
    except Exception as e:
        logging.error(e)


def transport_op(sess, saver, ckpt_state, FLAGS):
    tf.logging.info('started transport model...')
    export_dir = "dfs://ea119dfssearch1--cn-shanghai/pai/release/{}/{}/{}/data/".format(FLAGS.user_name,
                                                                                        FLAGS.model_name,
                                                                                        FLAGS.model_version)

    clear_dfs_dir(export_dir)

    # rtp restore meta file must with step id
    final_path = str(ckpt_state.model_checkpoint_path)
    step = int(final_path.split('-')[-1])
    saver.save(sess, export_dir, global_step=step)
    tf.logging.info('finished transport model...')

    tf.logging.info('started transport fg.json...')
    fg_json_path_save = "dfs://ea119dfssearch1--cn-shanghai/pai/release/{}/{}/{}/data/fg.json".format(
        FLAGS.user_name,
        FLAGS.model_name,
        FLAGS.model_version)

    with open(fg_tf_file_path, 'r') as f:
        feature_configs_tf = json.load(f)

    feature_configs = transform_fg(feature_configs_tf)
    for feature_config in feature_configs['features']:
        tf.logging.info("feature_config: {}".format(feature_config))

    with tf.afile.AFile(fg_json_path_save, "w") as f:
        json.dump(feature_configs, f)
    tf.logging.info('finished transport fg.json...')


def transport_op_prerank(sess, saver, ckpt_state, FLAGS):
    tf.logging.info('started transport model...')
    export_dir = "dfs://ea119dfssearch1--cn-shanghai/pai/release/{}/{}/{}/data/".format(FLAGS.user_name,
                                                                                        FLAGS.model_name,
                                                                                        FLAGS.model_version)

    # rtp restore meta file must with step id
    final_path = str(ckpt_state.model_checkpoint_path)
    step = int(final_path.split('-')[-1])
    saver.save(sess, export_dir, global_step=step)
    tf.logging.info('finished transport model...')

    tf.logging.info('started transport fg.json...')
    fg_json_path_save = "dfs://ea119dfssearch1--cn-shanghai/pai/release/{}/{}/{}/data/fg.json".format(
        FLAGS.user_name,
        FLAGS.model_name,
        FLAGS.model_version)

    with open(fg_tf_file_path, 'r') as f:
        feature_configs_tf = json.load(f)

    feature_configs = transform_fg(feature_configs_tf)
    mode = FLAGS.mode
    if mode == "transport":
        feature_configs = filter_fg_user(feature_configs)
    elif mode == "score_infer":
        feature_configs = filter_fg_score(feature_configs)

    for feature_config in feature_configs['features']:
        tf.logging.info("feature_config: {}".format(feature_config))

    with tf.afile.AFile(fg_json_path_save, "w") as f:
        json.dump(feature_configs, f)
    tf.logging.info('finished transport fg.json...')


def filter_fg_score(dic_0):
    names = ["user_vec", "item_vec"]
    save_list = []
    for dic_1 in dic_0['features']:
        if "feature_name" in dic_1 and dic_1["feature_name"] in names:
            save_list.append(dic_1)
    dic_0['features'] = save_list
    return dic_0


def filter_fg_user(dic_0):
    names = ["is_deep_f577", "is_deep_f578", "is_deep_f579", "is_deep_f580", "is_deep_f581", "is_deep_f582",
             "is_deep_f583", "is_deep_f584", "is_deep_f585", "is_deep_f586", "is_deep_f588", "is_deep_f590",
             "is_deep_f591", "is_deep_f594", "is_deep_f531", "triped_state", "triping_state", "go_state"]

    save_list = []
    for dic_1 in dic_0['features']:
        if "feature_name" in dic_1 and dic_1["feature_name"] in names:
            save_list.append(dic_1)
        elif "sequence_name" in dic_1:
            save_list.append(dic_1)
    dic_0['features'] = save_list
    return dic_0

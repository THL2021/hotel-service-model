# -*- coding: utf-8 -*-
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# train and dev
tf.flags.DEFINE_string('tables', '', '')
tf.flags.DEFINE_string('train_tables', '', '')
tf.flags.DEFINE_string('dev_tables', '', '')

# transfer
tf.flags.DEFINE_string('train_tables_source', '', '')
tf.flags.DEFINE_string('dev_tables_source', '', '')

tf.flags.DEFINE_string('model_dir', 'zy_model/lr/', 'model dir')
tf.flags.DEFINE_string('model_dir_restore', 'zy_model/lr/', 'model dir')
tf.flags.DEFINE_bool('is_sequence_train', False, 'flag for train mode')
tf.flags.DEFINE_bool('is_restore', False, 'flag for restore')

tf.flags.DEFINE_integer('batch_size', 512, 'batch size')
tf.flags.DEFINE_integer('max_train_step', 1000000, 'max_train_step')
tf.flags.DEFINE_integer('dev_total', 100000, 'dev_total')
tf.flags.DEFINE_bool('is_training', True, '')
tf.flags.DEFINE_float('keep_prob', 1.0, "")

tf.flags.DEFINE_string('buckets', "", 'buckets')
tf.flags.DEFINE_string('checkpointDir', "", 'checkpointDir')

# transport
tf.flags.DEFINE_string('user_name', "your_name", "")
tf.flags.DEFINE_integer('model_version', 1, "")
tf.flags.DEFINE_string('model_name', "ziya_supergul_intent", "")

# distribute
tf.flags.DEFINE_integer("task_index", None, "Worker task index")
tf.flags.DEFINE_string("ps_hosts", "", "ps hosts")
tf.flags.DEFINE_string("worker_hosts", "", "worker hosts")
tf.flags.DEFINE_string("job_name", None, "job name: worker or ps")
tf.flags.DEFINE_integer('aggregate', 100, 'aggregate batch number')
tf.flags.DEFINE_integer("save_time", 600, 'train epoch')

# output
tf.flags.DEFINE_string('output_tables', '', '')

# list
tf.flags.DEFINE_integer('list_size', 10, 'list_size')

# mode and model selection
tf.flags.DEFINE_string('mode', 'train', "train/dev/transport")
tf.flags.DEFINE_string('model', 'GLA', "model name")

FLAGS = tf.flags.FLAGS

# model selection and register
model_name = FLAGS.model
tf.logging.info("choose model: {}".format(model_name))

if model_name == 'ESMM':
    from models.ctcvr.ESMM import *
else:
    tf.logging.info("model name error: {}".format(model_name))
    raise ValueError


def main(_):
    if FLAGS.mode == 'train':
        distribute_train(FLAGS)
    elif FLAGS.mode == 'dev':
        dev(FLAGS)
    elif FLAGS.mode == 'transport':
        transport(FLAGS)
    elif FLAGS.mode == 'score_infer':
        score_infer(FLAGS)
    else:
        tf.logging.info("mode error: {}".format(FLAGS.mode))
        raise ValueError


if __name__ == '__main__':
    tf.app.run()

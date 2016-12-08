#!/usr/bin/env python3
import os

import tensorflow as tf

from read_data import read_babi, get_max_sizes
from data_utils import WordTable

flags = tf.app.flags

# directories
flags.DEFINE_string('model', 'Q2A', 'Model type [Q2A]')
flags.DEFINE_boolean('test', False, 'true for testing, false for training [False]')
flags.DEFINE_string('data_dir', 'babi', 'Data directory [babi]')
flags.DEFINE_string('save_dir', 'save', 'Save path [save]')
flags.DEFINE_string('load_dir', 'load', 'Load path [load]')

# training options
flags.DEFINE_bool('gpu', True, 'Use GPU? [True]')
flags.DEFINE_integer('batch_size', 128, 'Batch size during training and testing [128]')
flags.DEFINE_integer('num_epochs', 256, 'Number of epochs for training [256]')
flags.DEFINE_float('learning_rate', 0.002, 'Learning rate [0.002]')
flags.DEFINE_boolean('load', False, 'Start training from saved model? [False]')
flags.DEFINE_integer('acc_period', 10, 'Training accuracy display period [10]')
flags.DEFINE_integer('val_period', 40, 'Validation period (for display purpose) [40]')
flags.DEFINE_integer('save_period', 80, 'Save period [80]')

# model params
flags.DEFINE_integer('memory_step', 3, 'Episodic Memory steps [3]')
flags.DEFINE_string('memory_update', 'relu', 'Episodic meory update method - relu or gru [relu]')
flags.DEFINE_integer('embed_size', 80, 'Word embedding size [80]')
flags.DEFINE_integer('hidden_size', 80, 'Size of hidden units [80]')

# train hyperparameters
flags.DEFINE_float('weight_decay', 0.001, 'Weight decay - 0 to turn off L2 regularization [0.001]')
flags.DEFINE_float('keep_prob', 1., 'Dropout rate - 1.0 to turn off [1.0]')
flags.DEFINE_bool('batch_norm', True, 'Use batch normalization? [True]')

# bAbi dataset params
flags.DEFINE_integer('task', 1, 'bAbi Task number [1]')
flags.DEFINE_float('val_ratio', 0.1, 'Validation data ratio to training data [0.1]')

FLAGS = flags.FLAGS

def main(_):
    words = WordTable()
    if FLAGS.model == 'Q2A':
        from Q2A import DMN
    else:
        from A2Q import DMN

    train = read_babi(os.path.join(FLAGS.data_dir, 'train'), FLAGS.task, 'train', FLAGS.batch_size, words)
    test = read_babi(os.path.join(FLAGS.data_dir, 'test'), FLAGS.task, 'test', FLAGS.batch_size, words)
    val = train.split_dataset(FLAGS.val_ratio)

    FLAGS.max_sent_size, FLAGS.max_ques_size, FLAGS.max_fact_count = get_max_sizes(train, test, val)
    print('Word count: %d, Max sentence len : %d' % (words.vocab_size, FLAGS.max_sent_size))
    print('Max question len: %d, Max fact count: %d' % (FLAGS.max_ques_size, FLAGS.max_fact_count))

    # Modify save dir
    import time
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, 'task_{}_{}'.format(
        FLAGS.task, int(time.time())))

    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir, exist_ok=True)

    with tf.Session() as sess:
        model = DMN(FLAGS, words)
        sess.run(tf.initialize_all_variables())

        if FLAGS.test:
            model.load(sess)
            model.eval(sess, test, name='Test')
        else:
            if FLAGS.load:
                model.load(sess)
            model.train(sess, train, val)

if __name__ == '__main__':
    tf.app.run()

#!/usr/bin/env python3
import os
import sys
import tensorflow as tf
import time

from read_data import read_babi, get_max_sizes
from data_utils import WordTable

flags = tf.app.flags

# directories
flags.DEFINE_string('model', 'Q2A', 'Model type [Q2A]')
flags.DEFINE_string('mode', 'train', 'train or test or custom[train]')
flags.DEFINE_string('data_dir', 'babi', 'Data directory [babi]')
flags.DEFINE_string('save_dir', 'save', 'Save path [save]')
flags.DEFINE_string('load_dir', 'load', 'Load path [load]')
flags.DEFINE_string('expert_dir', '', 'Expert path []') ## for loading expert in GQ model

# training options
flags.DEFINE_bool('gpu', True, 'Use GPU? [True]')
flags.DEFINE_integer('batch_size', 128, 'Batch size during training and testing [128]')
flags.DEFINE_integer('num_epochs', 256, 'Number of epochs for training [256]')
flags.DEFINE_float('learning_rate', 0.002, 'Learning rate [0.002]')
flags.DEFINE_boolean('load', False, 'Start training from saved model? [False]')
flags.DEFINE_integer('acc_period', 10, 'Training accuracy display period [10]')
flags.DEFINE_integer('val_period', 40, 'Validation period (for display purpose) [40]')
flags.DEFINE_integer('save_period', 80, 'Save period [80]') ## not used, use val_period as save_period instead to perform 
                                                            ## early stopping

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
    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir, exist_ok=True)

    if FLAGS.model == 'Q2A':
        from Q2A import DMN as Model
    elif FLAGS.model == 'A2Q':
        from A2Q import DMN as Model
    elif FLAGS.model == 'SEQ2SEQ':
        from seq2seq import Seq2Seq as Model
    else:
        raise Exception("Unsupported model!")
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, FLAGS.model+'_task_{}'.format(
        FLAGS.task))

    train = read_babi(os.path.join(FLAGS.data_dir, 'train'), FLAGS.task, 'train', FLAGS.batch_size, words)
    test = read_babi(os.path.join(FLAGS.data_dir, 'test'), FLAGS.task, 'test', FLAGS.batch_size, words)
    val = train.split_dataset(FLAGS.val_ratio)
    FLAGS.max_sent_size, FLAGS.max_ques_size, FLAGS.max_fact_count = get_max_sizes(train, test, val)
    if FLAGS.mode == 'train':
        summary_dir = os.path.join(FLAGS.save_dir, "summary")
        if tf.gfile.Exists(summary_dir):
            tf.gfile.DeleteRecursively(summary_dir)
        model = Model(FLAGS, words)
        if FLAGS.load:
            model.load()

        model.train(train, val)
        model.save_flags()

    elif FLAGS.mode == 'test':
        model = Model(FLAGS, words)
        if FLAGS.load:
            model.load()
        else:
            print('Need Loading')
            return

        model.eval(test, name='Test')
        model.decode(test, sys.stdout, all=False)

    else:
        raise Exception("Unsupported executing mode!")

if __name__ == '__main__':
    tf.app.run()

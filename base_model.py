import sys

import numpy as np
import tensorflow as tf
from tqdm import tqdm

class BaseModel(object):
    """ Code from mem2nn-tensorflow. """
    def __init__(self, params, words):
        self.params = params
        self.words = words
        self.save_dir = params.save_dir
        self.load_dir = params.load_dir

        with tf.variable_scope('DMN'):
            print("Building DMN...")
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.build()
            self.merged = tf.merge_all_summaries()

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        self.summary_writer = tf.train.SummaryWriter(logdir=self.save_dir, graph=self.sess.graph)
    
    def __del__(self):
        self.sess.close()

    def build(self):
        raise NotImplementedError()

    def get_feed_dict(self, batch, is_train):
        raise NotImplementedError()

    def train_batch(self, batch):
        feed_dict = self.get_feed_dict(batch, is_train=True)
        return self.sess.run([self.merged, self.opt_op, self.global_step], feed_dict=feed_dict)

    def test_batch(self, batch):
        feed_dict = self.get_feed_dict(batch, is_train=False)
        return self.sess.run([self.total_loss, self.global_step], feed_dict=feed_dict)

    def train(self, train_data, val_data):
        params = self.params
        num_epochs = params.num_epochs
        num_batches = train_data.num_batches

        print("Training %d epochs ..." % num_epochs)
        for epoch_no in tqdm(range(num_epochs), desc='Epoch', maxinterval=86400, ncols=100):
            for _ in range(num_batches):
                batch = train_data.next_batch()
                summary, _, global_step = self.train_batch(batch)

            self.summary_writer.add_summary(summary, global_step)
            train_data.reset()

            if (epoch_no + 1) % params.acc_period == 0:
                print()  # Newline for TQDM
                self.eval(train_data, name='Training')

            if val_data and (epoch_no + 1) % params.val_period == 0:
                self.eval(val_data, name='Validation')

            if (epoch_no + 1) % params.save_period == 0:
                self.save()

        print("Training completed.")

    def eval(self, data, name):
        num_batches = data.num_batches
        losses = []
        for _ in range(num_batches):
            batch = data.next_batch()
            cur_loss, global_step = self.test_batch(batch)
            losses.append(cur_loss)
        data.reset()
        loss = np.mean(losses)

        print("[%s] step %d, Loss = %.4f" % \
              (name, global_step, loss))
        return loss

    def save(self):
        print("Saving model to dir %s" % self.save_dir)
        import os
        self.saver.save(self.sess, os.path.join(self.save_dir, 'run'), self.global_step)

    def load(self):
        print("Loading model ...")
        checkpoint = tf.train.get_checkpoint_state(self.load_dir)
        if checkpoint is None:
            print("Error: No saved model found. Please train first.")
            sys.exit(0)
        self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

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

        self.sess = tf.Session()
        with tf.variable_scope('DMN'):
            print("Building DMN...")
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.build()
            self.merged = tf.summary.merge_all()

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self.summary_writer = tf.summary.FileWriter(logdir=self.save_dir, graph=self.sess.graph)

        if hasattr(self, "accuracy"):
            self.eval_list = [self.total_loss, self.global_step, self.accuracy]
        else:
            self.eval_list = [self.total_loss, self.global_step]
    
    def __del__(self):
        self.sess.close()

    def build(self):
        raise NotImplementedError()

    def decode(self):
        raise NotImplementedError()

    def get_feed_dict(self, batch, is_train):
        raise NotImplementedError()

    def train_batch(self, batch):
        feed_dict = self.get_feed_dict(batch, is_train=True)
        return self.sess.run([self.merged, self.opt_op, self.global_step], feed_dict=feed_dict)

    def test_batch(self, batch):
        feed_dict = self.get_feed_dict(batch, is_train=False)
        return self.sess.run(self.eval_list, feed_dict=feed_dict)

    def train(self, train_data, val_data):
        params = self.params
        num_epochs = params.num_epochs
        num_batches = train_data.num_batches

        min_loss = np.inf
        print("Training %d epochs ..." % num_epochs)
        try:
            for epoch_no in tqdm(range(num_epochs), desc='Epoch', maxinterval=86400, ncols=100):
                for _ in range(num_batches):
                    batch = train_data.next_batch()
                    summary, _, global_step = self.train_batch(batch)

                self.summary_writer.add_summary(summary, global_step)
                train_data.reset()

                if (epoch_no + 1) % params.acc_period == 0:
                    tqdm.write("")  # Newline for TQDM
                    self.eval(train_data, name='Training')

                if (epoch_no + 1) % params.val_period == 0:
                    loss = np.inf
                    if val_data:
                        loss = self.eval(val_data, name='Validation')
                    if loss <= min_loss:
                        min_loss = loss
                        self.save()
            print("Training completed.")

        except KeyboardInterrupt:
            loss = np.inf
            if val_data:
                loss = self.eval(val_data, name='Validation')
            if loss <= min_loss:
                min_loss = loss
                self.save()
            print("Stop the training by console!")


    def eval(self, data, name):
        num_batches = data.num_batches
        losses = []
        accs = []
        for _ in range(num_batches):
            batch = data.next_batch()
            out = self.test_batch(batch)
            losses.append(out[0])
            global_step = out[1]
            if len(out) == 3:
                accs.append(out[2])
        data.reset()
        loss = np.mean(losses)
        if len(accs) == 0:
            acc = 0.
        else:
            acc = np.mean(accs)
        tqdm.write("[%s] step %d, Loss = %.4f, Acc = %.4f" % \
              (name, global_step, loss, acc))
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

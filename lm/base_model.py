import sys
import os
import json

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tqdm import tqdm


def load_params_dict(filename):
    with open(filename, 'r') as file:
        params_dict = json.load(file)
    return params_dict


class BaseModel(object):
    """ Code from mem2nn-tensorflow. """
    def __init__(self, vocab_size, params, *args):
        self.vocab_size = vocab_size
        ## dirs ##
        self.save_dir = params.save_dir
        self.load_dir = params.load_dir

        ## set params ##
        self.action = params.action
        self.params = params;
        self.num_steps = params.lm_num_steps
        ## build graph ##
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.params.gpu_fraction)
        self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options))
        with self.graph.as_default():
            default_init = xavier_initializer()
            with tf.variable_scope('Language_model', initializer=default_init):
                print("Building Language Model...")
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.min_validation_loss = tf.Variable(np.inf, name='validation_loss', trainable=False)
                self.new_validation_loss = tf.placeholder('float32', name='new_validation_loss');
                self.assign_min_validation_loss = self.min_validation_loss.assign(self.new_validation_loss).op
                if self.action == 'train':
                    self.build(eval_flag=False)
                elif self.action == 'test':
                    self.build(eval_flag=True)
                self.merged = tf.summary.merge_all()
                self.init_op = tf.global_variables_initializer()

        ## init saver ##
        with self.graph.as_default():
            self.saver = tf.train.Saver()

        ## init variables ##
        if not self.load_dir == '':
            print("Loading model ...")
            self.load()
        else:
            summary_dir = os.path.join(self.save_dir, "summary")
            if tf.gfile.Exists(summary_dir):
                tf.gfile.DeleteRecursively(summary_dir)
            print("Init model ...")
            self.sess.run(self.init_op)

        ## init saver ##
        if self.action == 'train':
            summary_dir = os.path.join(self.save_dir, "summary")
            self.summary_writer = tf.summary.FileWriter(logdir=summary_dir, graph=self.sess.graph)

        ## train & eval run output ##

    def __del__(self):
        if hasattr(self, "sess"):
            self.sess.close()

    def build(self):
        raise NotImplementedError()

    def get_feed_dict(self, batch, is_train):
        raise NotImplementedError()

    def save_params(self):
        raise NotImplementedError()

    def train_batch(self, feed_dict):
        train_list = [self.merged, self.opt_op, self.global_step, self.final_state, self.loss]
        return self.sess.run(train_list, feed_dict=feed_dict)

    def test_batch(self, feed_dict):
        eval_list = [self.loss, self.global_step, self.final_state, self.log_perp]
        return self.sess.run(eval_list, feed_dict=feed_dict)

    def pre_train(self, train_data, val_data):
        assert self.action == 'train'
        params = self.params
        num_epochs = params.lm_num_epoch

        min_loss = self.sess.run(self.min_validation_loss)
        print("Training %d epochs ..." % num_epochs)
        try:
            for epoch_no in range(num_epochs):
                ## training process ##
                ## fetch initial state ##
                iters = 0
                losses = 0.0
                for step, (x, y) in enumerate(train_data.get_batch()):
                    if step == 0:
                        init_feed_dict = {self.x: x}
                        state = self.sess.run(self.initial_state, feed_dict=init_feed_dict)
                    feed_dict = self.get_feed_dict((x, y), is_train=True, state=state)
                    summary, _, global_step, final_state, cur_loss = self.train_batch(feed_dict)
                    losses += cur_loss
                    iters += self.num_steps
                    if step % (train_data.epoch_size // 10) == 10:
                        print('[%d] perp: %f' % (global_step, np.exp(losses / iters)))
                    state = final_state
                self.summary_writer.add_summary(summary, global_step)
                ## evaluation process, including train and val ##
                if (epoch_no + 1) % params.lm_period == 0:
                    self.eval(train_data, name='Training')

                if (epoch_no + 1) % params.lm_val_period == 0:
                    loss = np.inf
                    if val_data:
                        loss , _, _ = self.eval(val_data, name='Validation')
                    if loss <= min_loss:
                        self.sess.run(self.assign_min_validation_loss, {self.new_validation_loss: loss})
                        min_loss = loss
                        self.save()
            print("Training completed.")

        except KeyboardInterrupt:
            loss = np.inf
            if val_data:
                loss, _, _ = self.eval(val_data, name='Validation')
            if loss <= min_loss:
                self.sess.run(self.assign_min_validation_loss, {self.new_validation_loss: loss})
                min_loss = loss
                self.save()
            print("Stop the training by console!")


    def eval(self, data, name):
        losses = 0.0
        iters = 0
        log_perps = []
        for step, (x, y) in enumerate(data.get_batch()):
            if step == 0:
                init_feed_dict = {self.x: x}
                state = self.sess.run(self.initial_state, feed_dict=init_feed_dict)
            feed_dict = self.get_feed_dict((x, y), is_train=False, state=state)
            batch_loss, global_step, state, log_perp = self.test_batch(feed_dict)
            losses += batch_loss
            iters += self.num_steps
            log_perps.append(log_perp)
        perp = np.exp(losses / iters)
        #print(np.exp(np.sum(np.array(log_perps).transpose(), axis = 1) / iters))
        log_perps = np.array(log_perps).transpose()
        #print("[%s] step %d, Perp = %.4f" % \
        #      (name, global_step, perp))
        return perp, log_perps, iters

    def save(self):
        assert self.action == 'train'
        print("[lm] Saving model to dir %s" % self.save_dir)
        self.saver.save(self.sess, os.path.join(self.save_dir, 'run'), self.global_step)

    def load(self):
        checkpoint = tf.train.get_checkpoint_state(self.load_dir)
        if checkpoint is None:
            print("Error: No saved model found. Please train first.")
            sys.exit(0)
        self.saver.restore(self.sess, checkpoint.model_checkpoint_path)


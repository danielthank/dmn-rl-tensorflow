import sys
import os
import json
from copy import deepcopy
from collections import namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tqdm import tqdm


def isnamedtupleinstance(x):
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple: return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple): return False
    return all(type(n)==str for n in f)


def load_flags(filename):
    with open(filename, 'r') as file:
        params = json.load(file)
    return params


def flags2params(flags):
    if isnamedtupleinstance(flags):
        return deepcopy(flags)
    elif isinstance(flags, type(tf.app.flags.FLAGS)):
        flags_dict = deepcopy(flags.__dict__['__flags'])
        subclass = namedtuple('subclass', flags_dict.keys())
        params = subclass(**flags_dict)
        return params
    else:
        raise Exception("Unsupported type of flags!")


class BaseModel(object):
    """ Code from mem2nn-tensorflow. """
    def __init__(self, flags, words):
        self.params = flags2params(flags)
        self.words = words
        self.mode = self.params.mode
        if self.mode == 'test':
            params_filename = os.path.join(self.params.save_dir, 'params.json')
            load_params = load_flags(params_filename)
            if not load_params['task'] == self.params.task:
                raise Exception("incompatible task!")
            self.params = self.params._replace(**load_params)

        self.sess = tf.Session()
        default_init = xavier_initializer()
        with tf.variable_scope('DMN', initializer=default_init):
            print("Building DMN...")
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            if self.mode == 'train':
                self.build(feed_previous=False, forward_only=False)
            elif self.mode == 'test':
                self.build(feed_previous=True, forward_only=True)
            self.merged = tf.summary.merge_all()

        if hasattr(self, "accuracy"):
            self.eval_list = [self.total_loss, self.global_step, self.accuracy]
        else:
            self.eval_list = [self.total_loss, self.global_step]

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self.load_dir = self.params.load_dir
        if self.mode == 'train':
            self.save_dir = self.params.save_dir
            summary_dir = os.path.join(self.save_dir, "summary")
            self.summary_writer = tf.summary.FileWriter(logdir=summary_dir, graph=self.sess.graph)
    
    def __del__(self):
        if hasattr(self, "sess"):
            self.sess.close()

    def build(self):
        raise NotImplementedError()

    def get_feed_dict(self, batch, is_train):
        raise NotImplementedError()

    def train_batch(self, feed_dict):
        return self.sess.run([self.merged, self.opt_op, self.global_step], feed_dict=feed_dict)

    def test_batch(self, feed_dict):
        return self.sess.run(self.eval_list, feed_dict=feed_dict)

    def reward_batch(self, batch, pred_qs):
        feed_dict = self.get_feed_dict(batch, is_train=False)
        feed_dict[self.q] = pred_qs
        output_probs = self.sess.run(self.output, feed_dict=feed_dict)
        assert output_probs.shape == (self.params.batch_size, self.words.vocab_size)
        return np.max(output_probs, axis=1)

    def train(self, train_data, val_data):
        assert self.mode == 'train'
        params = self.params
        num_epochs = params.num_epochs
        num_batches = train_data.num_batches

        min_loss = np.inf
        print("Training %d epochs ..." % num_epochs)
        try:
            for epoch_no in tqdm(range(num_epochs), desc='Epoch', maxinterval=86400, ncols=100):
                for _ in range(num_batches):
                    batch = train_data.next_batch()
                    feed_dict = self.get_feed_dict(batch, is_train=True)
                    summary, _, global_step = self.train_batch(feed_dict)

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
            feed_dict = self.get_feed_dict(batch, is_train=False)
            out = self.test_batch(feed_dict)
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
        assert self.mode == 'train'
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

    def save_flags(self):
        assert self.mode == 'train'
        params = self.params
        filename = os.path.join(params.save_dir, "params.json")
        save_params = {'memory_step': params.memory_step,
                       'memory_update': params.memory_update,
                       'embed_size': params.embed_size,
                       'hidden_size': params.hidden_size,
                       'weight_decay': params.weight_decay,
                       'keep_prob': params.keep_prob,
                       'batch_norm': params.batch_norm,
                       'task': params.task}
        with open(filename, 'w') as file:
            json.dump(save_params, file, indent=4)

    def decode(self, data, outputfile, all=True):
        tqdm.write("Write decoded output...")
        num_batches = data.num_batches
        for _ in range(num_batches):
            batch = data.next_batch()
            feed_dict = self.get_feed_dict(batch, False)
            outputs = self.sess.run(self.output, feed_dict=feed_dict)
            for idx in range(len(outputs)):
                content = "".join(token+' ' for sent in batch[0][idx] for token in sent)
                question = "".join(token+' ' for token in batch[1][idx])
                ans = batch[2][idx]
                p_ans = self.words.idx2word[np.argmax(outputs[idx])]
                outputfile.write("Content: "+content.strip()+'\n')
                outputfile.write("Question: "+question.strip()+'\n')
                outputfile.write("Ans: "+ans+'\n')
                outputfile.write("Predict_A: "+p_ans+"\n\n")
            if not all:
                break
        data.reset()
        tqdm.write("Finished")


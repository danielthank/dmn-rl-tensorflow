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
    def __init__(self, words, params, *args):
        ## words ##
        self.words = words

        ## dirs ##
        self.save_dir = params.save_dir
        self.load_dir = params.load_dir

        ## set params ##
        self.action = params.action
        self.params = params;

        ## build graph ##
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            default_init = xavier_initializer()
            with tf.variable_scope('Expert', initializer=default_init):
                print("Building Expert Model...")
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.min_validation_loss = tf.Variable(np.inf, name='validation_loss', trainable=False)
                self.new_validation_loss = tf.placeholder('float32', name='new_validation_loss');
                self.assign_min_validation_loss = self.min_validation_loss.assign(self.new_validation_loss).op
                if self.action == 'train':
                    self.build(forward_only=False)
                elif self.action == 'test':
                    self.build(forward_only=True)
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
        train_list = [self.merged, self.opt_op, self.global_step]
        return self.sess.run(train_list, feed_dict=feed_dict)

    def test_batch(self, feed_dict):
        eval_list = [self.total_loss, self.global_step, self.accuracy]
        return self.sess.run(eval_list, feed_dict=feed_dict)

    def output_by_question(self, batch, pred_qs):
        feed_dict = self.get_feed_dict(batch, is_train=False)
        feed_dict[self.q] = pred_qs
        """
        output_probs = self.sess.run(self.output, feed_dict=feed_dict)
        assert output_probs.shape == (pred_qs.shape[0], self.words.vocab_size)
        return output_probs
        """
        ans_logits = self.sess.run(self.ans_logits, feed_dict=feed_dict)
        assert ans_logits.shape == (pred_qs.shape[0], self.words.vocab_size)
        return ans_logits

    def train(self, train_data, val_data):
        assert self.action == 'train'
        params = self.params
        num_epochs = params.num_epochs
        num_batches = train_data.get_batch_num(full_batch=False)

        min_loss = self.sess.run(self.min_validation_loss)
        print("Training %d epochs ..." % num_epochs)
        try:
            for epoch_no in tqdm(range(num_epochs), desc='Epoch', maxinterval=86400, ncols=100):
                for _ in range(num_batches):
                    batch = train_data.next_batch(full_batch = False)
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
                        self.sess.run(self.assign_min_validation_loss, {self.new_validation_loss: loss})
                        min_loss = loss
                        self.save()
            print("Training completed.")

        except KeyboardInterrupt:
            loss = np.inf
            if val_data:
                loss = self.eval(val_data, name='Validation')
            if loss <= min_loss:
                self.sess.run(self.assign_min_validation_loss, {self.new_validation_loss: loss})
                min_loss = loss
                self.save()
            print("Stop the training by console!")


    def eval(self, data, name):
        num_batches = data.get_batch_num(full_batch = False)
        losses = []
        accs = []
        for _ in range(num_batches):
            batch = data.next_batch(full_batch = False)
            feed_dict = self.get_feed_dict(batch, is_train=False)
            batch_loss, global_step, batch_acc = self.test_batch(feed_dict)
            losses.append(batch_loss)
            accs.append(batch_acc)
        data.reset()
        loss = np.mean(losses)
        acc = np.mean(accs)
        
        if name == 'Training':
            self.train_acc = acc
        elif name == 'Validation':
            self.val_acc = acc
        elif name == 'Test':
            self.test_acc = acc

        tqdm.write("[%s] step %d, Loss = %.4f, Acc = %.4f" % \
              (name, global_step, loss, acc))
        return loss

    def save(self):
        assert self.action == 'train'
        print("Saving model to dir %s" % self.save_dir)
        self.saver.save(self.sess, os.path.join(self.save_dir, 'run'), self.global_step)

    def load(self):
        checkpoint = tf.train.get_checkpoint_state(self.load_dir)
        if checkpoint is None:
            print("Error: No saved model found. Please train first.")
            sys.exit(0)
        self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

    def _get_good_output(self, sent):
        output = ''
        for token in sent:
            if token == 0:
                break
            output = output + self.words.idx2word[token] + ' '
        return output.strip()

    def decode(self, data, outputfile, inputfile, all=True):
        tqdm.write("Write decoded output...")
        num_batches = data.num_batches
        for _ in range(num_batches):
            batch = data.next_batch()
            feed_dict = self.get_feed_dict(batch, is_train=False)
            outputs = self.sess.run(self.output, feed_dict=feed_dict)
            for idx in range(len(batch[0])):
                print("Content:", file=outputfile)
                for i, sent in enumerate(batch[0][idx]):
                    if sent[0] == 0:
                        break
                    print(i+1, self._get_good_output(sent), file=outputfile)

                print("Question:", self._get_good_output(batch[1][idx]), file=outputfile)
                ans = self.words.idx2word[batch[2][idx]]
                print("Ans: ", ans, file=outputfile)
                p_ans = self.words.idx2word[np.argmax(outputs[idx])]
                print("Predict_A:", p_ans, file=outputfile)
                print(file=outputfile)

            if not all:
                break
        data.reset()
        tqdm.write("Finished")

        for i, word in enumerate(self.words.idx2word):
            print(i, word)

        from data_helper.read_data import tokenize
        while True:
            story = np.zeros((1, self.params.story_size, self.params.sentence_size), dtype='int32')
            question = np.zeros((1, self.params.question_size), dtype='int32')
            answer = np.zeros((1,),  dtype='int32')
            sentence_cnt = 0;
            for line in inputfile:
                tokens = tokenize(line)
                if '?' in tokens:
                    word_cnt = 0
                    for token in tokens:
                        if token in self.words.word2idx:
                            question[0][word_cnt] = self.words.word2idx[token]
                            word_cnt = word_cnt + 1
                    break

                word_cnt = 0
                for token in tokens:
                    if token in self.words.word2idx:
                        story[0][sentence_cnt][word_cnt] = self.words.word2idx[token]
                        word_cnt = word_cnt + 1
                if word_cnt:
                    sentence_cnt = sentence_cnt + 1;

            batch = (story, question, answer)
            print("Content:", file=outputfile)
            for i, sent in enumerate(batch[0][0]):
                if sent[0] == 0:
                    break
                print(i+1, self._get_good_output(sent), file=outputfile)
            print("Question:", self._get_good_output(batch[1][0]), file=outputfile)
            feed_dict = self.get_feed_dict((story, question, answer), is_train=False)
            #outputs = self.sess.run(self.output, feed_dict=feed_dict)
            outputs = self.sess.run(self.ans_logits, feed_dict=feed_dict)
            p_ans = self.words.idx2word[np.argmax(outputs[0])]
            order = np.argsort(outputs[0])[::-1]
            print("Predict_A:", file=outputfile)
            for i in range(5):
                print(self.words.idx2word[order[i]], outputs[0][order[i]])
            print(file=outputfile)


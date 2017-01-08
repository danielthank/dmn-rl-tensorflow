import sys
import os
import json
from copy import deepcopy
from collections import namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tqdm import tqdm

from expert.dmn import DMN as EXPERT_DMN
from expert.ren import REN as EXPERT_REN


## accessible expert model ##
EXPERT_MODEL = {'expert_dmn': EXPERT_DMN,
                'expert_ren': EXPERT_REN}


class BaseModel(object):
    """ Code from mem2nn-tensorflow. """
    def __init__(self, words, params, expert_params, *args):
        ## words ##
        self.words = words

        ## dirs ##
        self.save_dir = params.save_dir
        self.load_dir = params.load_dir

        ## set params ##
        self.action = params.action
        self.params = params

        ## build model graph ##
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            with tf.variable_scope('Learner', initializer=xavier_initializer()):
                print("Building Learner model...")
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
            self.load()
        else:
            summary_dir = os.path.join(self.save_dir, "summary")
            if tf.gfile.Exists(summary_dir):
                tf.gfile.DeleteRecursively(summary_dir)
            self.sess.run(self.init_op)

        ## summary writer##
        if self.action == 'train':
            summary_dir = os.path.join(self.save_dir, "summary")
            self.summary_writer = tf.summary.FileWriter(logdir=summary_dir, graph=self.sess.graph)

        ## train & eval run output ##
        self.train_list = [self.merged, self.opt_op, self.global_step] 
        self.eval_list = [self.total_loss, self.global_step]
        
        ## load expert ##
        if expert_params == None:
            raise Exception("Need expert params to load an expert!")
        else:
            print("Loading Expert...")
            self.expert = self.load_expert(expert_params)

    
    def __del__(self):
        if hasattr(self, "sess"):
            self.sess.close()

    def build(self):
        raise NotImplementedError()

    def get_feed_dict(self, batch, is_train):
        raise NotImplementedError()

    def save_params(self):
        raise NotImplementedError()

    def get_question(self, feed_dict):
        outputs = self.sess.run(self.output, feed_dict=feed_dict)
        outputs = np.argmax(np.stack(outputs, axis=1), axis=2)
        return outputs

    def train_batch(self, feed_dict):
        return self.sess.run(self.train_list, feed_dict=feed_dict)

    def test_batch(self, feed_dict):
        return self.sess.run(self.eval_list, feed_dict=feed_dict)

    def train(self, train_data, val_data):
        params = self.params
        assert self.action == 'train'
        num_epochs = params.num_epochs
        num_batches = train_data.num_batches

        min_loss = self.sess.run(self.min_validation_loss)
        print("Training %d epochs ..." % num_epochs)
        try:
            for epoch_no in tqdm(range(num_epochs), desc='Epoch', maxinterval=86400, ncols=100):
                for _ in range(num_batches):
                    batch = train_data.next_batch()
                    feed_dict = self.get_feed_dict(batch, feed_previous=False, is_train=True)
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
        num_batches = data.num_batches
        losses = []
        for _ in range(num_batches):
            batch = data.next_batch()
            feed_dict = self.get_feed_dict(batch, feed_previous=True, is_train=False)
            batch_loss, global_step = self.test_batch(feed_dict)
            losses.append(batch_loss)
        data.reset()
        loss = np.mean(losses)
        tqdm.write("[%s] step %d, Loss = %.4f" % \
              (name, global_step, loss))
        return loss

    def save(self):
        assert self.action == 'train'
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

    def load_expert(self, expert_params):
        expert_model_name = '{}_{}'.format(expert_params.target, expert_params.arch)
        ExpertModel = EXPERT_MODEL[expert_model_name]
        return ExpertModel(self.words, expert_params, None)

    def ask_expert(self, batch, pred_qs):
        output_probs = self.expert.output_by_question(batch, pred_qs)
        max_index = np.argmax(output_probs, axis=1)
        inverse_entropy = np.sum(np.log(output_probs + 1e-20) * output_probs, axis=1)
        return inverse_entropy, max_index

    def decode(self, data, outputfile, all=True):
        tqdm.write("Write decoded output...")
        num_batches = data.num_batches
        for _ in range(num_batches):
            batch = data.next_batch()
            feed_dict = self.get_feed_dict(batch, feed_previous=True, is_train=False)
            outputs = self.get_question(feed_dict)
            expert_entropys, expert_anses = self.ask_expert(batch, outputs)
            for idx, output in enumerate(outputs):
                """
                pred_q = []
                for time in output:
                    pred_q.append(self.words.idx2word[time])
                """
                content = "".join(self.words.idx2word[token]+' ' for sent in batch[0][idx] for token in sent)
                question = "".join(self.words.idx2word[token]+' ' for token in batch[1][idx])
                ans = batch[2][idx]
                pred_q = "".join(self.words.idx2word[token]+' ' for token in output)
                expert_entropy = expert_entropys[idx]
                expert_ans = self.words.idx2word[expert_anses[idx]]
                outputfile.write("Content: "+content.strip()+'\n')
                outputfile.write("Question: "+question.strip()+'\n')
                outputfile.write("Ans: "+self.words.idx2word[ans]+'\n')
                outputfile.write("Predict_Q: "+pred_q.strip()+"\n")
                outputfile.write("Expert Result: "+str(expert_entropy)+"\t"+expert_ans+"\n\n")
            if not all:
                break
        data.reset()
        tqdm.write("Finished")

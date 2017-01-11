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
        if params.action == 'train':
            self.summary_dir = os.path.join(self.save_dir, 'pretrain_summary')
            self.train_batch = self.pre_train_batch
        elif params.action == 'rl':
            self.summary_dir = os.path.join(self.save_dir, 'RL_summary')
            self.train_batch = self.rl_train_batch

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
                if not self.action == 'test':
                    self.build(forward_only=False)
                else:
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
            if tf.gfile.Exists(self.summary_dir):
                tf.gfile.DeleteRecursively(self.summary_dir)
            print("Init model ...")
            self.sess.run(self.init_op)

        ## summary writer##
        if not self.action == 'test':
            self.summary_writer = tf.summary.FileWriter(logdir=self.summary_dir, graph=self.sess.graph)

        ## train & eval run output ##
        if self.action == 'train':
            self.train_list = [self.merged, self.Pre_opt_op, self.global_step]
        elif self.action == 'rl':
            self.train_list = [self.merged, self.RL_opt_op, self.global_step]
        if not self.action == 'rl':
            self.eval_list = [self.total_loss, self.global_step]
            self.test_batch = self.pre_test_batch
        else:
            self.eval_list = [self.J, self.global_step]
            self.test_batch = self.rl_test_batch
        
        ## load expert ##
        assert not expert_params == None
        print("Loading Expert...")
        self.expert = self.load_expert(expert_params)

        self.baseline = 0.

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
        assert outputs.shape == (self.params.batch_size, self.params.question_size)
        return outputs

    def get_rewards(self, batch, q_outputs):
        expert_entropys, expert_anses = self.ask_expert(batch, q_outputs)
        CQ_rewards = self.CQ_reward(batch[0], q_outputs)
        tot_rewards = CQ_rewards#+0.*(np.exp(expert_entropys)-0.5)
        return tot_rewards

    def pre_train_batch(self, batch):
        feed_dict = self.get_feed_dict(batch, feed_previous=False, is_train=True)
        return self.sess.run(self.train_list, feed_dict=feed_dict)

    def rl_train_batch(self, batch):
        A = self.words.vocab_size
        feed_dict = self.get_feed_dict(batch, feed_previous=True, is_train=True)
        q_outputs = self.get_question(feed_dict)

        rewards = self.get_rewards(batch, q_outputs)
        chosen_one_hot = (np.arange(A) == q_outputs[:, :, None]).astype('float32')

        feed_dict.update({self.chosen_one_hot: chosen_one_hot, self.rewards: (rewards - self.baseline)})
        return self.sess.run(self.train_list, feed_dict=feed_dict), np.mean(rewards)

    def pre_test_batch(self, batch):
        feed_dict = self.get_feed_dict(batch, feed_previous=True, is_train=False)
        return self.sess.run(self.eval_list, feed_dict=feed_dict)

    def rl_test_batch(self, batch):
        A = self.words.vocab_size
        feed_dict = self.get_feed_dict(batch, feed_previous=True, is_train=False)
        q_outputs = self.get_question(feed_dict)

        rewards = self.get_rewards(batch, q_outputs)
        chosen_one_hot = (np.arange(A) == q_outputs[:, :, None]).astype('float32')

        feed_dict.update({self.chosen_one_hot: chosen_one_hot, self.rewards: (rewards - self.baseline)})
        return self.sess.run(self.eval_list, feed_dict=feed_dict)

    def train(self, train_data, val_data):
        params = self.params
        assert not self.action == 'test'
        num_epochs = params.num_epochs
        num_batches = train_data.num_batches

        min_loss = self.sess.run(self.min_validation_loss)
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

    def rl_train(self, train_data, val_data):
        params = self.params
        assert not self.action == 'test'
        num_epochs = params.num_epochs
        num_batches = train_data.num_batches

        min_loss = self.sess.run(self.min_validation_loss)
        r = 0.
        print("Training %d epochs ..." % num_epochs)
        try:
            for epoch_no in tqdm(range(num_epochs), desc='Epoch', maxinterval=86400, ncols=100):
                for _ in range(num_batches):
                    batch = train_data.next_batch()
                    (summary, _, global_step), mean_r = self.train_batch(batch)
                    r += mean_r
                self.baseline = 0.9*self.baseline + 0.1*r/num_batches if not self.baseline == 0. else r/num_batches

                self.summary_writer.add_summary(summary, global_step)
                train_data.reset()

                if (epoch_no + 1) % params.acc_period == 0:
                    tqdm.write("")  # Newline for TQDM
                    self.eval(train_data, name='Training')
                    tqdm.write("rewards: "+str(r))

                if (epoch_no + 1) % params.val_period == 0:
                    loss = np.inf
                    if val_data:
                        loss = self.eval(val_data, name='Validation')
                    if loss <= min_loss:
                        self.sess.run(self.assign_min_validation_loss, {self.new_validation_loss: loss})
                        min_loss = loss
                        self.save()
                r = 0.
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
            batch_loss, global_step = self.test_batch(batch)
            losses.append(batch_loss)
        data.reset()
        loss = np.mean(losses)
        tqdm.write("[%s] step %d, Loss = %.4f" % \
              (name, global_step, loss))
        return loss

    def save(self):
        assert not self.action == 'test'
        print("Saving model to dir %s" % self.save_dir)
        import os
        self.saver.save(self.sess, os.path.join(self.save_dir, 'run'), self.global_step)

    def load(self):
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

    def CQ_similarity(self, content, keyterm):
        ## keyterm should be a word idx
        sent_cnt = 0.
        content_merge = []
        for sent in content:
            if sent[0] == 0:
                break
            content_merge.extend(sent)
            sent_cnt += 1.
        cnt = content_merge.count(keyterm)
        if cnt == 0.:
            return -1.
        else:
            return cnt / sent_cnt

    def CQ_reward(self, batch_x, qs_idxs):
        rewards = []
        for b, q_idxs in enumerate(qs_idxs):
            keyterm = self.words.find_keyterm_by_idx(*q_idxs)
            CQ_sim = self.CQ_similarity(batch_x[b], keyterm)
            rewards.append(CQ_sim)
        return np.array(rewards).astype('float32')

    def decode(self, data, outputfile, all=True):
        tqdm.write("Write decoded output...")
        num_batches = data.num_batches
        for _ in range(num_batches):
            batch = data.next_batch()
            feed_dict = self.get_feed_dict(batch, feed_previous=True, is_train=False)
            outputs = self.get_question(feed_dict)
            expert_entropys, expert_anses = self.ask_expert(batch, outputs)
            for idx, output in enumerate(outputs):
                pred_q = [self.words.idx2word[token] for token in output]
                keyterm = self.words.find_keyterm_by_idx(*output)
                CQ_sim = self.CQ_similarity(batch[0][idx], keyterm)

                content = "".join(self.words.idx2word[token]+' ' for sent in batch[0][idx] for token in sent)
                question = "".join(self.words.idx2word[token]+' ' for token in batch[1][idx])
                ans = self.words.idx2word[batch[2][idx]]
                pred_q = "".join(token+' ' for token in pred_q)
                expert_entropy = expert_entropys[idx]
                expert_ans = self.words.idx2word[expert_anses[idx]]
                outputfile.write("Content: "+content.strip()+'\n')
                outputfile.write("Question: "+question.strip()+'\n')
                outputfile.write("Ans: "+ans+'\n')
                outputfile.write("Predict_Q: "+pred_q.strip()+"\tKeyTerm: "+self.words.idx2word[keyterm]+"\tCount: "+str(CQ_sim)+'\n')
                outputfile.write("Expert Entropy: "+str(expert_entropy)+'\t'+expert_ans+"\n\n")
            if not all:
                break
        data.reset()
        tqdm.write("Finished")

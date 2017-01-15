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
        elif params.action == 'rl':
            self.summary_dir = os.path.join(self.save_dir, 'RL_summary')

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
                self.merged_PRE = tf.summary.merge_all(key='PRE_SUMM')
                self.merged_QA = tf.summary.merge_all(key='QA_SUMM')
                self.merged_RL = tf.summary.merge_all(key='RL_SUMM')
                self.merged_VAR = tf.summary.merge_all(key='VAR_SUMM')
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

    def get_feed_dict(self, batch, feed_previous, is_train):
        raise NotImplementedError()

    def save_params(self):
        raise NotImplementedError()

    def get_question(self, feed_dict):
        q_probs = self.sess.run(self.q_probs, feed_dict=feed_dict)
        q_idxes = np.argmax(np.stack(q_probs, axis=1), axis=2)
        assert q_idxes.shape == (self.params.batch_size, self.params.question_size)
        return q_idxes

    def get_rewards(self, batch, pred_qs):
        expert_entropys, expert_anses = self.ask_expert(batch, pred_qs)
        learner_entropys, learner_anses = self.ask_learner(batch, pred_qs)
        CQ_rewards = self.CQ_reward(batch[0], pred_qs)
        tot_rewards = CQ_rewards#+0.*(np.exp(expert_entropys)-0.5)
        #tot_rewards = np.exp(expert_entropys) - np.exp(learner_entropys)
        return tot_rewards, expert_anses

    def pre_train_batch(self, batch):
        pre_train_list = [self.merged_PRE, self.Pre_opt_op, self.global_step]
        feed_dict = self.get_feed_dict(batch, feed_previous=False, is_train=True)
        return self.sess.run(pre_train_list, feed_dict=feed_dict)

    def QA_train_batch(self, batch):
        QA_train_list = [self.merged_QA, self.QA_opt_op, self.global_step, self.QA_total_loss, self.accuracy]
        feed_dict = self.get_feed_dict(batch, feed_previous=True, is_train=True)
        return self.sess.run(QA_train_list, feed_dict=feed_dict)

    def rl_train_batch(self, batch):
        A = self.words.vocab_size
        feed_dict = self.get_feed_dict(batch, feed_previous=True, is_train=True)
        pred_qs = self.get_question(feed_dict)

        rewards, expert_anses = self.get_rewards(batch, pred_qs)
        chosen_one_hot = (np.arange(A) == pred_qs[:, :, None]).astype('float32')

        rl_train_list = [self.merged_RL, self.RL_opt_op, self.global_step, self.J]
        feed_dict.update({self.chosen_one_hot: chosen_one_hot, self.rewards: (rewards - self.baseline)})
        return self.sess.run(rl_train_list, feed_dict=feed_dict), np.mean(rewards), pred_qs, expert_anses

    def pre_test_batch(self, batch):
        pre_eval_list = [self.QA_total_loss, self.QG_total_loss, self.accuracy, self.global_step]
        feed_dict = self.get_feed_dict(batch, feed_previous=True, is_train=False)
        return self.sess.run(pre_eval_list, feed_dict=feed_dict)

    def QA_test_batch(self, batch):
        QA_eval_list = [self.QA_total_loss, self.global_step, self.accuracy]
        feed_dict = self.get_feed_dict(batch, feed_previous=True, is_train=False)
        return self.sess.run(QA_eval_list, feed_dict=feed_dict)

    def rl_test_batch(self, batch):
        A = self.words.vocab_size
        feed_dict = self.get_feed_dict(batch, feed_previous=True, is_train=False)
        pred_qs = self.get_question(feed_dict)

        rewards, expert_anses = self.get_rewards(batch, pred_qs)
        chosen_one_hot = (np.arange(A) == pred_qs[:, :, None]).astype('float32')

        rl_eval_list = [self.J, self.QA_total_loss, self.accuracy, self.global_step]
        feed_dict.update({self.chosen_one_hot: chosen_one_hot, self.rewards: (rewards - self.baseline)})
        return self.sess.run(rl_eval_list, feed_dict=feed_dict), np.mean(rewards)

    def train(self, train_data, val_data):
        params = self.params
        assert not self.action == 'test'
        num_epochs = params.num_epochs
        num_batches = train_data.num_batches

        print("Pre-Training on 100 samples")
        batch = train_data.get_batch_cnt(100)
        try:
            for epoch_no in tqdm(range(num_epochs), desc='Epoch', maxinterval=86400, ncols=100):
                summ,  _, global_step = self.pre_train_batch(batch)
                QA_loss, QG_loss, acc, global_step = self.pre_test_batch(batch)
                var_summ = self.sess.run(self.merged_VAR)
                tqdm.write("[Training] step {:d}, QA_Loss = {:.4f}, QG_Loss = {:.4f}, ACC = {:.4f}".format(global_step, QA_loss, QG_loss, acc))
                self.summary_writer.add_summary(summ, global_step)
                self.summary_writer.add_summary(var_summ, global_step)
        except KeyboardInterrupt:
            print("Stop the training by console!")
        self.save()
        print('complete')

    def rl_train(self, train_data, val_data):
        params = self.params
        assert not self.action == 'test'
        num_epochs = params.num_epochs
        num_batches = train_data.num_batches

        min_val_loss = self.sess.run(self.min_validation_loss)
        print("RL Training %d epochs ..." % num_epochs)
        try:
            for epoch_no in tqdm(range(num_epochs), desc='Epoch', maxinterval=86400, ncols=100):
                r = []
                tot_J = []
                QA_batch_x = np.empty((0, params.story_size, params.sentence_size), dtype='int32')
                QA_batch_q = np.empty((0, params.question_size), dtype='int32')
                QA_batch_y = np.empty((0, ), dtype='int32')
                tot_QA_loss = []
                tot_QA_acc = []
                for i in range(num_batches):
                    batch = train_data.next_batch()
                    (rl_summ, _, global_step, J), mean_r, pred_qs, expert_anses = self.rl_train_batch(batch)
                    r.append(mean_r)
                    tot_J.append(J)
                    QA_batch_x = np.concatenate((QA_batch_x, batch[0]), axis=0)
                    QA_batch_q = np.concatenate((QA_batch_q, pred_qs), axis=0)
                    QA_batch_y = np.concatenate((QA_batch_y, expert_anses), axis=0)
                    if i % 3 == 0:
                        QA_summ, _, global_step, QA_loss, acc = self.QA_train_batch((QA_batch_x, QA_batch_q, QA_batch_y))
                        tot_QA_loss.append(QA_loss)
                        tot_QA_acc.append(acc)
                        QA_batch_x = np.empty((0, params.story_size, params.sentence_size), dtype='int32')
                        QA_batch_q = np.empty((0, params.question_size), dtype='int32')
                        QA_batch_y = np.empty((0, ), dtype='int32')

                self.baseline = 0.9*self.baseline + 0.1*np.mean(r) if not self.baseline == 0. else np.mean(r)

                var_summ = self.sess.run(self.merged_VAR)
                self.summary_writer.add_summary(rl_summ, global_step)
                self.summary_writer.add_summary(QA_summ, global_step)
                self.summary_writer.add_summary(var_summ, global_step)
                train_data.reset()

                if (epoch_no + 1) % params.acc_period == 0:
                    tqdm.write("")  # Newline for TQDM
                    write_line = "[Training] " + \
                                 "step {:d} ".format(global_step) + \
                                 "J = {:.4f} ".format(np.mean(tot_J)) + \
                                 "reward = {:.4f} ".format(np.mean(r)) + \
                                 "QA_Loss = {:.4f} ".format(np.mean(tot_QA_loss)) + \
                                 "QA_ACC = {:.4f}".format(np.mean(tot_QA_acc))
                    tqdm.write(write_line)

                if (epoch_no + 1) % params.val_period == 0:
                    val_loss = np.inf
                    if val_data:
                        val_loss = self.eval(val_data, name='Validation')
                    if val_loss <= min_val_loss:
                        self.sess.run(self.assign_min_validation_loss, {self.new_validation_loss: val_loss})
                        min_val_loss = val_loss
                        self.save()
            print("Training completed.")

        except KeyboardInterrupt:
            val_loss = np.inf
            if val_data:
                val_loss = self.eval(val_data, name='Validation')
            if val_loss <= min_val_loss:
                self.sess.run(self.assign_min_validation_loss, {self.new_validation_loss: val_loss})
                min_val_loss = val_loss
                self.save()
            print("Stop the training by console!")

    def eval(self, data, name):
        num_batches = data.num_batches
        tot_J = []
        tot_QA_loss = []
        tot_QA_acc = []
        r = []
        for _ in range(num_batches):
            batch = data.next_batch()
            (J, QA_loss, QA_acc, global_step), mean_r = self.rl_test_batch(batch)
            tot_J.append(J)
            tot_QA_loss.append(QA_loss)
            tot_QA_acc.append(QA_acc)
            r.append(mean_r)
        data.reset()
        avg_J = np.mean(tot_J)
        avg_r = np.mean(r)
        avg_QA_loss = np.mean(tot_QA_loss)
        avg_QA_acc = np.mean(tot_QA_acc)
        write_line = ("[%s] " % name) + \
                     "step {:d} ".format(global_step) + \
                     "J = {:.4f} ".format(avg_J) + \
                     "reward = {:.4f} ".format(avg_r) + \
                     "QA_Loss = {:.4f} ".format(avg_QA_loss) + \
                     "QA_ACC = {:.4f}".format(avg_QA_acc)
        tqdm.write(write_line)
        return avg_QA_loss

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

    def ask_learner(self, batch, pred_qs):
        feed_dict = self.get_feed_dict(batch, feed_previous=False, is_train=False)
        feed_dict[self.q] = pred_qs
        ans_probs = self.sess.run(self.QA_ans, feed_dict=feed_dict)
        assert ans_probs.shape == (self.params.batch_size, self.words.vocab_size)
        max_index = np.argmax(ans_probs, axis=1)
        inverse_entropy = np.sum(np.log(ans_probs + 1e-20) * ans_probs, axis=1)
        return inverse_entropy, max_index

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

    def decode(self, data, outputfile, inputfile, all=True):
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

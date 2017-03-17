import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import static_bidirectional_rnn

from learner.base_model import BaseModel
from dmn_helper.nn import weight, bias, dropout, batch_norm, variable_summary, gumbel_softmax
from dmn_helper.episode_module import EpisodeModule
from ren_helper.model_utils import get_sequence_length


EPS = 1e-20
OPTIMIZER_SUMMARIES = ["learning_rate",
                       "loss",
                       "gradients",
                       "gradient_norm"]


class Seq2Seq(BaseModel):
    def build(self, forward_only):
        params = self.params
        L, Q, F = params.sentence_size, params.question_size, params.story_size
        V, d, A = params.dmn_embedding_size, params.dmn_embedding_size, self.words.vocab_size

        # initialize self
        # placeholders
        input = tf.placeholder('int32', shape=[None, F, L], name='x')
        question = tf.placeholder('int32', shape=[None, Q], name='q')
        answer = tf.placeholder('int32', shape=[None], name='y')
        self.batch_size = tf.shape(answer)[0]
        fact_counts = get_sequence_length(input)
        is_training = tf.placeholder(tf.bool)
        feed_previous = tf.placeholder(tf.bool)

        # Prepare parameters
        gru = rnn.GRUCell(d)
        l = self.positional_encoding()

        with tf.variable_scope('Embedding'):
            embedding = weight('embedding', [A, V], init='uniform', range=3**(1/2))
            embedding_mask = tf.constant([0 if i == 0 else 1 for i in range(A)], dtype=tf.float32, shape=[A, 1])
            embedding = embedding * embedding_mask
            variable_summary([embedding])

        with tf.name_scope('Tied_SentenceReader'):
            input_embed = tf.nn.embedding_lookup(embedding, input) # [batch, story, sentence] -> [batch, story, sentence, embedding_size]
            # apply positional encoding
            encoded = l * input_embed
            facts = tf.reduce_sum(encoded, 2)  # [batch, story, embedding_size]
            # dropout
            # facts = dropout(facts, params.dmn_keep_prob, is_training)

        with tf.variable_scope('Question_embed'):
            ques_embed = tf.nn.embedding_lookup(embedding, question) # [N, Q, V]
        
        with tf.variable_scope('InputFusion') as scope:
            (output_fw, output_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(gru,
                                                                             gru,
                                                                             facts,
                                                                             sequence_length=fact_counts,
                                                                             dtype=tf.float32)
            f_outputs = output_fw + output_bw
            q_init_state = state_fw + state_bw

        ## QA Branch
        with tf.variable_scope('QA_branch'):
            QA_ans_logits = self.QA_branch(gru, ques_embed, f_outputs, is_training)
            QA_ans = tf.nn.softmax(QA_ans_logits)

            with tf.name_scope('QA_Loss'):
                # Cross-Entropy loss
                QA_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=QA_ans_logits, labels=answer)
                QA_loss = tf.reduce_mean(QA_cross_entropy)
                QA_total_loss = QA_loss + params.dmn_weight_decay * tf.add_n(tf.get_collection('l2'))

            with tf.variable_scope('QA_Accuracy'):
                # Accuracy
                predicts = tf.cast(tf.argmax(QA_ans_logits, 1), 'int32')
                corrects = tf.equal(predicts, answer)
                num_corrects = tf.reduce_sum(tf.cast(corrects, tf.float32))
                accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

        ## QG Branch
        with tf.variable_scope('QG_branch') as scope:
            q_logprobs = self.QG_branch(embedding, ques_embed, q_init_state, feed_previous)
            q_probs = [tf.nn.softmax(out) for out in q_logprobs]
            variables = [v for v in tf.trainable_variables() if v.name.startswith(scope.name)]
            variable_summary(variables)

            with tf.name_scope('QG_Loss'):
                target_list = tf.unstack(tf.transpose(question)) # Q * [N]
                # Cross-Entropy loss
                QG_loss = tf.contrib.legacy_seq2seq.sequence_loss(q_logprobs, target_list,
                                                   [tf.ones(shape=tf.stack([tf.shape(answer)[0],]), dtype=tf.float32)] * Q )
                QG_total_loss = QG_loss + params.dmn_weight_decay * tf.add_n(tf.get_collection('l2'))

        # Policy Gradient
        chosen_one_hot = tf.placeholder(tf.float32, shape=[None, Q, A], name='act')
        rewards = tf.placeholder(tf.float32, shape=[None], name='rewards')

        with tf.name_scope("PolicyGradient"):
            stack_q_probs = tf.stack(q_probs, axis=1) # Q * [N, A] -> [N, Q, A]
            act_probs = stack_q_probs * chosen_one_hot # [N, Q, A]
            act_probs = tf.reduce_prod(tf.reduce_sum(act_probs, axis=2), axis=1) # [N, Q, A] -> [N, Q] -> [N]

            J = -1.*tf.reduce_mean(tf.log(act_probs+EPS)*rewards) + params.dmn_weight_decay*tf.add_n(tf.get_collection('l2'))


        # placeholders
        self.x = input
        self.q = question
        self.y = answer
        self.fc = fact_counts
        self.is_training = is_training
        self.feed_previous = feed_previous

        # policy gradient placeholders
        self.chosen_one_hot = chosen_one_hot
        self.rewards = rewards

        # QA output tensors
        self.QA_ans = QA_ans
        self.QA_total_loss = QA_total_loss
        self.num_corrects = num_corrects
        self.accuracy = accuracy

        # QG output tensors
        self.q_probs = q_probs
        self.QG_total_loss = QG_total_loss

        # policy gradient output tensors
        self.J = J

        # optimizer ops
        if not forward_only:
            self.RL_opt_op, RL_opt_sc = self.RLOpt()
            self.Pre_opt_op, Pre_opt_sc = self.PreOpt()
            self.QA_opt_op, QA_opt_sc = self.QAOpt()

    def QA_branch(self, gru, ques_embed, f_outputs, is_training):
        params = self.params
        sentence_size, question_size, story_size = params.sentence_size, params.question_size, params.story_size
        vocab_size = self.words.vocab_size
        embedding_size = params.dmn_embedding_size
        hidden_size = params.dmn_embedding_size

        decoder_input = tf.unstack(ques_embed, axis=1)
        init_state = tf.zeros(shape=[self.batch_size, hidden_size], dtype=tf.float32)
 
        _, state = tf.contrib.legacy_seq2seq.attention_decoder(decoder_input,
                                                               init_state,
                                                               f_outputs,
                                                               gru)
        W = tf.get_variable('hidden2vocab', shape=[hidden_size, vocab_size])
        ans_logits = tf.matmul(state, W)
        return ans_logits

    def QG_branch(self, embedding, ques_embed, q_init_state, feed_previous):
        params = self.params
        L, Q, F = params.sentence_size, params.question_size, params.story_size
        V, d, A = params.dmn_embedding_size, params.dmn_embedding_size, self.words.vocab_size
        ## output projection weight ##
        proj_w = weight('proj_w', [d, A])
        proj_b = bias('proj_b', A)
        ## build decoder inputs ##
        go_pad = tf.ones(tf.stack([self.batch_size, 1]), dtype=tf.int32)
        go_pad = tf.nn.embedding_lookup(embedding, go_pad) # [N, 1, V]
        decoder_inputs = tf.concat(axis=1, values=[go_pad, ques_embed]) # [N, Q+1, V]
        decoder_inputs = tf.unstack(decoder_inputs, axis=1)[:-1] # Q * [N, V]
        ## question module rnn cell ##
        q_cell = rnn.GRUCell(d)
        ## decoder loop function ##
        def _loop_fn(prev, i):
            prev = tf.matmul(prev, proj_w) + proj_b
            prev_symbol = tf.argmax(prev, 1)
            emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
            return emb_prev
        ## decoder ##
        def decoder(feed_previous_bool):
            loop_function = _loop_fn if feed_previous_bool else None
            reuse = None if feed_previous_bool else True
            with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                q_outputs, _ = tf.contrib.legacy_seq2seq.rnn_decoder(decoder_inputs=decoder_inputs,
                                                         initial_state=q_init_state,
                                                         cell=q_cell,
                                                         loop_function=loop_function)
                return q_outputs
        q_outputs = tf.cond(feed_previous,
                            lambda: decoder(True),
                            lambda: decoder(False))
        q_logprobs = [tf.matmul(out, proj_w) + proj_b for out in q_outputs]

        return q_logprobs

    def PreOpt(self):
        with tf.variable_scope("PreOpt") as scope:
            def learning_rate_decay_fn(lr, global_step):
                return tf.train.exponential_decay(lr,
                                                  global_step,
                                                  decay_steps=100,
                                                  decay_rate=0.95,
                                                  staircase=True)
            OPTIMIZER_SUMMARIES = ["learning_rate",
                                   "loss"]
            opt_op = tf.contrib.layers.optimize_loss(0.5*self.QA_total_loss + 0.5*self.QG_total_loss,
                                                     self.global_step,
                                                     learning_rate=self.params.learning_rate,
                                                     optimizer=tf.train.AdamOptimizer,
                                                     clip_gradients=1.,
                                                     learning_rate_decay_fn=learning_rate_decay_fn,
                                                     summaries=OPTIMIZER_SUMMARIES)
            for var in tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope.name):
                tf.add_to_collection("PRE_SUMM", var)
            return opt_op, scope.name

    def QAOpt(self):
        with tf.variable_scope("QAOpt") as scope:
            def learning_rate_decay_fn(lr, global_step):
                return tf.train.exponential_decay(lr,
                                                  global_step,
                                                  decay_steps=5000,
                                                  decay_rate=0.95,
                                                  staircase=True)
            OPTIMIZER_SUMMARIES = ["learning_rate",
                                   "loss"]
            opt_op = tf.contrib.layers.optimize_loss(self.QA_total_loss,
                                                     self.global_step,
                                                     learning_rate=self.params.learning_rate,
                                                     optimizer=tf.train.AdamOptimizer,
                                                     clip_gradients=5.,
                                                     learning_rate_decay_fn=learning_rate_decay_fn,
                                                     summaries=OPTIMIZER_SUMMARIES)
            for var in tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope.name):
                tf.add_to_collection("QA_SUMM", var)
            return opt_op, scope.name

    def RLOpt(self):
        with tf.variable_scope("RLOpt") as scope:
            def learning_rate_decay_fn(lr, global_step):
                return tf.train.exponential_decay(lr,
                                                  global_step,
                                                  decay_steps=5000,
                                                  decay_rate=0.95,
                                                  staircase=True)
            OPTIMIZER_SUMMARIES = ["learning_rate",
                                   "loss"]
            opt_op = tf.contrib.layers.optimize_loss(self.J,
                                                     self.global_step,
                                                     learning_rate=self.params.learning_rate,
                                                     optimizer=tf.train.AdamOptimizer,
                                                     clip_gradients=5.,
                                                     learning_rate_decay_fn=learning_rate_decay_fn,
                                                     summaries=OPTIMIZER_SUMMARIES)
            for var in tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope.name):
                tf.add_to_collection("RL_SUMM", var)
            return opt_op, scope.name

    def positional_encoding(self):
        V, L = self.params.dmn_embedding_size, self.params.sentence_size
        encoding = np.zeros([L, V])
        for l in range(L):
            for v in range(V):
                encoding[l, v] = (1 - float(l)/L) - (float(v)/V)*(1 - 2.0*l/L)

        return encoding

    def get_feed_dict(self, batches, feed_previous, is_train):
        return {
            self.x: batches[0],
            self.q: batches[1],
            self.y: batches[2],
            self.is_training: is_train,
            self.feed_previous: feed_previous
        }

    def save_params(self):
        assert not self.action == 'test'
        params = self.params
        filename = os.path.join(self.save_dir, "params.json")
        save_params_dict = {'dmn_memory_step': params.dmn_memory_step,
                            'dmn_memory_update': params.dmn_memory_update,
                            'dmn_embedding_size': params.dmn_embedding_size,
                            'dmn_weight_decay': params.dmn_weight_decay,
                            'dmn_keep_prob': params.dmn_keep_prob,
                            'dmn_batch_norm': params.dmn_batch_norm,
                            'target': params.target,
                            'arch': params.arch,
                            'task': params.task}
        with open(filename, 'w') as file:
            json.dump(save_params_dict, file, indent=4)

import os
import json
import numpy as np
import tensorflow as tf
#from tensorflow.nn.rnn_cell import MultiRNNCell, GRUCell
from tensorflow.contrib.layers import fully_connected, variance_scaling_initializer
from tensorflow.python.layers.core import Dense

from learner.base_model import BaseModel
from tf_helper.nn import weight, bias, dropout, batch_norm, variable_summary
from tf_helper.nn import gumbel_softmax, attention_decoder, create_opt
from tf_helper.model_utils import get_sequence_length, positional_encoding


EPS = 1e-20
MultiRNNCell = tf.nn.rnn_cell.MultiRNNCell
GRUCell = tf.nn.rnn_cell.GRUCell


class Seq2Seq(BaseModel):
    def build(self, forward_only):
        params = self.params
        L, Q, F = params.sentence_size, params.question_size, params.story_size
        V, A = params.seq2seq_hidden_size, self.words.vocab_size
        N = params.batch_size

        input = tf.placeholder('int32', shape=[N, F, L], name='x')
        question = tf.placeholder('int32', shape=[N, Q], name='q')
        answer = tf.placeholder('int32', shape=[N], name='y')
        #self.batch_size = tf.shape(answer)[0]
        self.batch_size = N
        fact_counts = get_sequence_length(input)
        self.is_training = tf.placeholder(tf.bool)
        self.is_sample = tf.placeholder(tf.bool)
        feed_previous = tf.placeholder(tf.bool)
        story_positional_encoding = positional_encoding(L, V)
        question_positional_encoding = positional_encoding(Q, V)
        embedding_mask = tf.constant([0 if i == 0 else 1 for i in range(A)], dtype=tf.float32, shape=[A, 1])

        D_labels = tf.placeholder('bool', shape=[N], name='D_labels')

        with tf.variable_scope('QA', initializer=tf.contrib.layers.xavier_initializer()) as QA_scope:
            with tf.device("/cpu:0"):
                qa_embedding = tf.get_variable('qa_embedding', [A, V], initializer=tf.contrib.layers.xavier_initializer())
            qa_embedding = qa_embedding * embedding_mask
            with tf.variable_scope('SentenceReader'):
                with tf.device("/cpu:0"):
                    qa_story = tf.nn.embedding_lookup(qa_embedding, input) # [batch, story, sentence] -> [batch, story, sentence, embedding_size]
                # apply positional encoding
                qa_story = story_positional_encoding * qa_story
                qa_story = tf.reduce_sum(qa_story, 2)  # [batch, story, embedding_size]
                qa_story = dropout(qa_story, 0.5, self.is_training)
                num_layers = 3
                q_cell_fw = MultiRNNCell([GRUCell(V) for l in range(num_layers)])
                q_cell_bw = MultiRNNCell([GRUCell(V) for l in range(num_layers)])
                (qa_states_fw, qa_states_bw), (tmp, _) = tf.nn.bidirectional_dynamic_rnn(q_cell_fw,
                                                                                       q_cell_bw,
                                                                                       qa_story,
                                                                                       sequence_length=fact_counts,
                                                                                       dtype=tf.float32)
                qa_story = qa_states_fw + qa_states_bw
            with tf.name_scope('QuestionReader'):
                with tf.device("/cpu:0"):
                    qa_q = tf.nn.embedding_lookup(qa_embedding, question) # [N, Q, V]
                qa_q = question_positional_encoding * qa_q
                qa_q = tf.reduce_sum(qa_q, 1) # [N, V]
                qa_q = dropout(qa_q, 0.5, self.is_training)

            QA_ans_logits = self.QA_branch(qa_embedding, qa_q, qa_story)
            QA_ans = tf.nn.softmax(QA_ans_logits)
            QA_vars = [v for v in tf.trainable_variables() if v.name.startswith(QA_scope.name)]
            #variable_summary(variables)
            with tf.name_scope('Loss'):
                # Cross-Entropy loss
                QA_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=QA_ans_logits, labels=answer)
                QA_loss = tf.reduce_mean(QA_cross_entropy)
                #QA_total_loss = QA_loss + params.seq2seq_weight_decay * tf.add_n(tf.get_collection('l2'))
                QA_total_loss = QA_loss
            with tf.variable_scope('Accuracy'):
                # Accuracy
                predicts = tf.cast(tf.argmax(QA_ans_logits, 1), 'int32')
                corrects = tf.equal(predicts, answer)
                num_corrects = tf.reduce_sum(tf.cast(corrects, tf.float32))
                QA_accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))
            tf.summary.scalar('loss', QA_total_loss, collections=["QA_SUMM"])
            tf.summary.scalar('accuracy', QA_accuracy, collections=["QA_SUMM"])

        with tf.variable_scope('QG', initializer=tf.contrib.layers.xavier_initializer()):
            with tf.device("/cpu:0"):
                qg_embedding = tf.get_variable('qg_embedding', [A, V], initializer=tf.contrib.layers.xavier_initializer())
            qg_embedding = qg_embedding * embedding_mask
            with tf.variable_scope('SentenceReader'):
                with tf.device("/cpu:0"):
                    qg_story = tf.nn.embedding_lookup(qg_embedding, input) # [batch, story, sentence] -> [batch, story, sentence, embedding_size]
                # apply positional encoding
                qg_story = story_positional_encoding * qg_story
                qg_story = tf.reduce_sum(qg_story, 2)  # [batch, story, embedding_size]
                qg_story = dropout(qg_story, 0.5, self.is_training)
                num_layers = 3
                q_cell_fw = MultiRNNCell([GRUCell(V) for l in range(num_layers)])
                q_cell_bw = MultiRNNCell([GRUCell(V) for l in range(num_layers)])
                (qg_states_fw, qg_states_bw), (_, _) = tf.nn.bidirectional_dynamic_rnn(q_cell_fw,
                                                                                       q_cell_bw,
                                                                                       qg_story,
                                                                                       sequence_length=fact_counts,
                                                                                       dtype=tf.float32)
                qg_story = qg_states_fw + qg_states_bw
            with tf.name_scope('QuestionReader'):
                with tf.device("/cpu:0"):
                    qg_q = tf.nn.embedding_lookup(qg_embedding, question) # [N, Q, V]
                qg_q = dropout(qg_q, 0.5, self.is_training)
            q_logprobs, q_length, chosen_idxs = self.QG_branch(qg_embedding, qg_q, qg_story, feed_previous, self.is_training)
            q_probs = tf.nn.softmax(q_logprobs, dim=-1)
            with tf.name_scope('Loss'):
                target_list = tf.unstack(tf.transpose(question)) # Q * [N]
                # Cross-Entropy loss
                QG_loss = tf.contrib.seq2seq.sequence_loss(q_logprobs, question, tf.cast(tf.sequence_mask(q_length, Q),
                                                                                         tf.float32))
                                                   # [tf.ones(shape=tf.stack([tf.shape(answer)[0],]), dtype=tf.float32)] * Q )
                #QG_total_loss = QG_loss + params.seq2seq_weight_decay * tf.add_n(tf.get_collection('l2'))
                QG_total_loss = QG_loss
            tf.summary.scalar('loss', QG_total_loss, collections=["QG_SUMM"])

        # Policy Gradient
        with tf.name_scope("PolicyGradient"):
            chosen_one_hot = tf.placeholder(tf.float32, shape=[N, Q, A], name='act')
            rewards = tf.placeholder(tf.float32, shape=[N], name='rewards')
            baseline_t = tf.placeholder(tf.float32, shape=[], name='baseline')
            advantages = rewards - baseline_t
            tf.summary.scalar('rewards', tf.reduce_mean(rewards), collections=["RL_SUMM"])
            tf.summary.scalar('advantages', tf.reduce_mean(advantages), collections=["RL_SUMM"])
            # stack_q_probs = tf.transpose(q_probs, perm=[1, 0, 2]) # [Q , N, A] -> [N, Q, A]
            act_probs = q_probs * chosen_one_hot # [N, Q, A]
            act_probs = tf.reduce_prod(tf.reduce_sum(act_probs, axis=2), axis=1) # [N, Q, A] -> [N, Q] -> [N]
            # J = -1.*tf.reduce_mean(tf.log(act_probs+EPS)*advantages) + params.seq2seq_weight_decay*tf.add_n(tf.get_collection('l2'))
            J = -1.*tf.reduce_mean(tf.log(act_probs+EPS)*advantages)
            tf.summary.scalar('J', J, collections=["RL_SUMM"])

        # placeholders
        self.x = input
        self.q = question
        self.y = answer
        self.fc = fact_counts
        self.feed_previous = feed_previous

        self.D_labels = D_labels

        # policy gradient placeholders
        self.chosen_one_hot = chosen_one_hot
        self.rewards = rewards
        self.baseline_t = baseline_t

        # QA output tensors
        self.QA_ans_logits = QA_ans_logits
        self.QA_ans = QA_ans
        self.QA_init_op = tf.variables_initializer(var_list=QA_vars, name="QA_init_op")
        self.QA_total_loss = QA_total_loss
        self.num_corrects = num_corrects
        self.QA_accuracy = QA_accuracy

        # QG output tensors
        self.q_probs = q_probs
        self.chosen_idxs = chosen_idxs
        self.QG_total_loss = QG_total_loss

        self.J = J

        # optimizer ops
        if not forward_only:
            rl_l_rate = self.params.rl_learning_rate
            l_rate = self.params.learning_rate
            self.RL_opt_op = create_opt('RL_opt', self.J, rl_l_rate, self.RL_global_step, decay_steps=5000, clip=0.2)
            self.Pre_opt_op = create_opt('Pre_opt',
                                         0.5*self.QA_total_loss + 0.5*self.QG_total_loss,
                                         l_rate,
                                         self.Pre_global_step)
            self.QA_opt_op = create_opt('QA_opt', self.QA_total_loss, l_rate, self.QA_global_step)
            self.reQA_opt_op = create_opt('reQA_opt', self.QA_total_loss, l_rate, self.reQA_global_step)

        # merged summary ops
        self.merged_QA = tf.summary.merge_all(key='QA_SUMM')
        self.merged_QG = tf.summary.merge_all(key='QG_SUMM')
        self.merged_RL = tf.summary.merge_all(key='RL_SUMM')
        self.merged_VAR = tf.summary.merge_all(key='VAR_SUMM')

    def QA_branch(self, embedding, qa_q, qa_story):
        params = self.params
        go_pad = tf.ones(tf.stack([self.batch_size]), dtype=tf.int32)
        with tf.device("/cpu:0"):
            go_pad = tf.nn.embedding_lookup(embedding, go_pad) # [N, 1, V]
        num_layers = 2
        attn_cell = tf.contrib.seq2seq.AttentionWrapper(
            GRUCell(params.seq2seq_hidden_size),
            tf.contrib.seq2seq.LuongAttention(params.seq2seq_hidden_size, qa_story)
        )
        basic_cell = GRUCell(params.seq2seq_hidden_size)
        multi_cell = MultiRNNCell([attn_cell, basic_cell])

        initial_state = (attn_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size).clone(cell_state=qa_q),
                         basic_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size))

        output, _ = multi_cell(go_pad, initial_state)
        q_logprobs = tf.layers.dense(output, self.words.vocab_size)
        q_logprobs = tf.contrib.layers.batch_norm(q_logprobs, decay=0.9, is_training=self.is_training, center=True, scale=True,
                                                     updates_collections=None, scope='BatchNorm')

        return q_logprobs

    def QG_branch(self, embedding, qg_q, qg_story, feed_previous, is_training):
        params = self.params
        L, Q, F = params.sentence_size, params.question_size, params.story_size
        V, A = params.seq2seq_hidden_size, self.words.vocab_size
        ## build decoder inputs ##
        go_pad = tf.ones(tf.stack([self.batch_size, 1]), dtype=tf.int32)
        with tf.device("/cpu:0"):
            go_pad = tf.nn.embedding_lookup(embedding, go_pad) # [N, 1, V]
        decoder_inputs = tf.concat(axis=1, values=[go_pad, qg_q])[:,:-1] # [N, Q, V]
        # decoder_inputs = tf.unstack(decoder_inputs, axis=1)[:-1] # Q * [N, V]

        ## output idxs ##
        chosen_idxs = []

        PRE_EPS = tf.constant(0.5, dtype='float32')
        INIT_EPS = tf.constant(0.01, dtype='float32')
        FIN_EPS = tf.constant(0.01, dtype='float32')
        EXPLORE = tf.constant(25e3, dtype='float32')
        f32_RL_step = tf.cast(self.RL_global_step, 'float32')
        f32_Pre_step = tf.cast(self.Pre_global_step, 'float32')
        explore_eps =  tf.case({f32_Pre_step > f32_RL_step: (lambda: PRE_EPS),
                                EXPLORE <= (f32_RL_step - f32_Pre_step): (lambda: FIN_EPS)},
                               default=(lambda: INIT_EPS - (INIT_EPS - FIN_EPS) * (f32_RL_step - f32_Pre_step) / EXPLORE),
                               exclusive=True)
        # explore_eps = tf.cond(self.is_sample, explore_eps, tf.constant(0))
        tf.summary.scalar("explore_eps", explore_eps, collections=["VAR_SUMM"])
        ## question module rnn cell ##
        attn_cell = tf.contrib.seq2seq.AttentionWrapper(
            GRUCell(params.seq2seq_hidden_size),
            tf.contrib.seq2seq.LuongAttention(params.seq2seq_hidden_size, qg_story)
        )
        basic_cell = GRUCell(params.seq2seq_hidden_size)
        multi_cell = MultiRNNCell([attn_cell, basic_cell])
        decoder_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
            inputs=decoder_inputs,
            sequence_length=tf.constant([params.question_size] * params.batch_size),
            embedding=embedding,
            sampling_probability=tf.constant(0.0)
        )

        initial_state = (attn_cell.zero_state(dtype=tf.float32, batch_size=params.batch_size).clone(cell_state=qg_story[:, -1]),
                         basic_cell.zero_state(dtype=tf.float32, batch_size=params.batch_size))

        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=multi_cell,
            helper=decoder_helper,
            initial_state=initial_state,
            output_layer=Dense(self.words.vocab_size)
        )
        final_outputs, final_state, final_length = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder
        )
        q_logprobs = final_outputs.rnn_output
        chosen_idxs = final_outputs.sample_id
        # q_logprobs = tf.contrib.layers.batch_norm(final_state, decay=0.9, is_training=self.is_training, center=True, scale=True,
                                                  # updates_collections=None, scope='BatchNorm')

        # assert len(chosen_idxs) == Q
        return q_logprobs, final_length, chosen_idxs

    def def_run_list(self):
        self.pre_train_list = [self.merged_QA, self.merged_QG, self.Pre_global_step, self.Pre_opt_op]
        self.QA_train_list  = [self.merged_QA, self.QA_global_step, self.QA_opt_op]
        self.rl_train_list  = [self.merged_RL, self.RL_global_step, self.RL_opt_op]
        #self.rl_train_list  = [self.merged_RL, self.RL_global_step, self.RL_global_step]
        #self.D_train_list   = [self.merged_D, self.D_global_step, self.D_opt_op]
        self.pre_test_list  = [self.merged_QA, self.merged_QG, self.Pre_global_step,
                               0.5*self.QA_total_loss+0.5*self.QG_total_loss]
        self.rl_test_list   = [self.merged_QA, self.merged_QG, self.QA_global_step, self.RL_global_step, self.QG_total_loss]

        self.reQA_train_list  = [self.merged_QA, self.reQA_global_step+self.QA_global_step+1000, self.reQA_opt_op]
        self.QA_test_list   = [self.merged_QA, self.reQA_global_step+self.QA_global_step+1000, self.QA_total_loss, self.QA_accuracy]

    def get_feed_dict(self, batches, feed_previous, is_train, is_sample):
        return {
            self.x: batches[0],
            self.q: batches[1],
            self.y: batches[2],
            self.is_training: is_train,
            self.is_sample: is_sample,
            self.feed_previous: feed_previous
        }

    def save_params(self):
        assert not self.action == 'test'
        params = self.params
        filename = os.path.join(self.save_dir, "params.json")
        save_params_dict = {'seq2seq_hidden_size': params.seq2seq_hidden_size,
                            'seq2seq_weight_decay': params.seq2seq_weight_decay,
                            'target': params.target,
                            'arch': params.arch,
                            'task': params.task}
        with open(filename, 'w') as file:
            json.dump(save_params_dict, file, indent=4)

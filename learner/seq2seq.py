import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import static_bidirectional_rnn
from tensorflow.contrib.layers import fully_connected, variance_scaling_initializer

from learner.base_model import BaseModel
from tf_helper.nn import weight, bias, dropout, batch_norm, variable_summary
from tf_helper.nn import gumbel_softmax, attention_decoder, create_opt
from tf_helper.model_utils import get_sequence_length, positional_encoding


EPS = 1e-20


class Seq2Seq(BaseModel):
    def build(self, forward_only):
        params = self.params
        L, Q, F = params.sentence_size, params.question_size, params.story_size
        V, A = params.seq2seq_hidden_size, self.words.vocab_size

        input = tf.placeholder('int32', shape=[None, F, L], name='x')
        question = tf.placeholder('int32', shape=[None, Q], name='q')
        answer = tf.placeholder('int32', shape=[None], name='y')
        self.batch_size = tf.shape(answer)[0]
        fact_counts = get_sequence_length(input)
        self.is_training = tf.placeholder(tf.bool)
        self.is_sample = tf.placeholder(tf.bool)
        feed_previous = tf.placeholder(tf.bool)
        story_positional_encoding = positional_encoding(L, V)
        question_positional_encoding = positional_encoding(Q, V)
        embedding_mask = tf.constant([0 if i == 0 else 1 for i in range(A)], dtype=tf.float32, shape=[A, 1])

        D_labels = tf.placeholder('bool', shape=[None], name='D_labels')

        with tf.variable_scope('QA', initializer=tf.contrib.layers.xavier_initializer()) as QA_scope:
            qa_embedding = tf.get_variable('qa_embedding', [A, V], initializer=tf.contrib.layers.xavier_initializer())
            qa_embedding = qa_embedding * embedding_mask
            with tf.variable_scope('SentenceReader'):
                qa_story = tf.nn.embedding_lookup(qa_embedding, input) # [batch, story, sentence] -> [batch, story, sentence, embedding_size]
                # apply positional encoding
                qa_story = story_positional_encoding * qa_story
                qa_story = tf.reduce_sum(qa_story, 2)  # [batch, story, embedding_size]
                qa_story = dropout(qa_story, 0.5, self.is_training)
                num_layers = 2
                q_cell_fw = tf.contrib.rnn.MultiRNNCell([rnn.LSTMCell(V) for l in range(num_layers)])
                q_cell_bw = tf.contrib.rnn.MultiRNNCell([rnn.LSTMCell(V) for l in range(num_layers)])
                (qa_states_fw, qa_states_bw), (_, _) = tf.nn.bidirectional_dynamic_rnn(q_cell_fw,
                                                                                       q_cell_bw,
                                                                                       qa_story,
                                                                                       sequence_length=fact_counts,
                                                                                       dtype=tf.float32)
                qa_story = qa_states_fw + qa_states_bw
            with tf.name_scope('QuestionReader'):
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
            qg_embedding = tf.get_variable('qg_embedding', [A, V], initializer=tf.contrib.layers.xavier_initializer())
            qg_embedding = qg_embedding * embedding_mask
            with tf.variable_scope('SentenceReader'):
                qg_story = tf.nn.embedding_lookup(qg_embedding, input) # [batch, story, sentence] -> [batch, story, sentence, embedding_size]
                # apply positional encoding
                qg_story = story_positional_encoding * qg_story
                qg_story = tf.reduce_sum(qg_story, 2)  # [batch, story, embedding_size]
                qg_story = dropout(qg_story, 0.5, self.is_training)
                num_layers = 2
                q_cell_fw = tf.contrib.rnn.MultiRNNCell([rnn.LSTMCell(V) for l in range(num_layers)])
                q_cell_bw = tf.contrib.rnn.MultiRNNCell([rnn.LSTMCell(V) for l in range(num_layers)])
                (qg_states_fw, qg_states_bw), (_, _) = tf.nn.bidirectional_dynamic_rnn(q_cell_fw,
                                                                                       q_cell_bw,
                                                                                       qg_story,
                                                                                       sequence_length=fact_counts,
                                                                                       dtype=tf.float32)
                qg_story = qg_states_fw + qg_states_bw
            with tf.name_scope('QuestionReader'):
                qg_q = tf.nn.embedding_lookup(qg_embedding, question) # [N, Q, V]
                qg_q = dropout(qg_q, 0.5, self.is_training)
            q_logprobs, chosen_idxs = self.QG_branch(qg_embedding, qg_q, qg_story, feed_previous, self.is_training)
            q_probs = tf.nn.softmax(q_logprobs, dim=-1)
            # variables = [v for v in tf.trainable_variables() if v.name.startswith(scope.name)]
            #variable_summary(variables)
            with tf.name_scope('Loss'):
                target_list = tf.unstack(tf.transpose(question)) # Q * [N]
                # Cross-Entropy loss
                QG_loss = tf.contrib.legacy_seq2seq.sequence_loss(tf.unstack(q_logprobs), target_list,
                                                   [tf.ones(shape=tf.stack([tf.shape(answer)[0],]), dtype=tf.float32)] * Q )
                #QG_total_loss = QG_loss + params.seq2seq_weight_decay * tf.add_n(tf.get_collection('l2'))
                QG_total_loss = QG_loss
            tf.summary.scalar('loss', QG_total_loss, collections=["QG_SUMM"])

        with tf.variable_scope("Discriminator", initializer=tf.contrib.layers.xavier_initializer()):
            D_embedding = tf.get_variable('D_embedding', [A, V], initializer=tf.contrib.layers.xavier_initializer())
            D_embedding = D_embedding * embedding_mask
            D_logits = self.Discriminator(D_embedding, question)
            D_probs = tf.nn.softmax(D_logits)[:, 1]
            #variables = [v for v in tf.trainable_variables() if v.name.startswith(scope.name)]
            #variable_summary(variables)
            # cross entropy loss
            #re_D_labels = tf.cast(tf.reshape(D_labels, shape=[-1, 1]), tf.float32)
            re_D_labels = tf.cast(D_labels, 'int32')
            #D_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits, labels=re_D_labels)
            D_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=D_logits, labels=re_D_labels)
            D_loss = tf.reduce_mean(D_cross_entropy)
            D_total_loss = D_loss
            # accuracy
            #D_predicts = tf.greater_equal(D_probs, 0.5)
            D_predicts = tf.cast(tf.argmax(D_logits, 1), 'int32')
            D_corrects = tf.equal(D_predicts, re_D_labels)
            D_accuracy = tf.reduce_mean(tf.cast(D_corrects, tf.float32))
            tf.summary.scalar('loss', D_total_loss, collections=["D_SUMM"])
            tf.summary.scalar('acc', D_accuracy, collections=["D_SUMM"])

        # Policy Gradient
        with tf.name_scope("PolicyGradient"):
            chosen_one_hot = tf.placeholder(tf.float32, shape=[None, Q, A], name='act')
            rewards = tf.placeholder(tf.float32, shape=[None], name='rewards')
            baseline_t = tf.placeholder(tf.float32, shape=[], name='baseline')
            advantages = rewards - baseline_t
            tf.summary.scalar('rewards', tf.reduce_mean(rewards), collections=["RL_SUMM"])
            tf.summary.scalar('advantages', tf.reduce_mean(advantages), collections=["RL_SUMM"])
            stack_q_probs = tf.transpose(q_probs, perm=[1, 0, 2]) # [Q , N, A] -> [N, Q, A]
            act_probs = stack_q_probs * chosen_one_hot # [N, Q, A]
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

        # Dirsciminator output tensors
        self.D_probs = D_probs
        self.D_total_loss = D_total_loss
        self.D_accuracy = D_accuracy

        # policy gradient output tensors
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
            self.D_opt_op = create_opt('D_opt', self.D_total_loss, l_rate*10, self.D_global_step)

        # merged summary ops
        self.merged_D = tf.summary.merge_all(key='D_SUMM')
        self.merged_QA = tf.summary.merge_all(key='QA_SUMM')
        self.merged_QG = tf.summary.merge_all(key='QG_SUMM')
        self.merged_RL = tf.summary.merge_all(key='RL_SUMM')
        self.merged_VAR = tf.summary.merge_all(key='VAR_SUMM')

    def Discriminator(self, D_embedding, question):
        params = self.params 
        Q = params.question_size
        V = params.seq2seq_hidden_size
        filter_sizes = [1, 3]
        num_filters = [3, 5]
        
        with tf.name_scope('D_QuestionReader'):
            D_q = tf.nn.embedding_lookup(D_embedding, question) # [N, Q, V]
            #D_q = tf.expand_dims(D_q, -1) # [N, Q, V, 1]
            D_qf = tf.reshape(D_q, shape=[-1, Q*V]) # [N, Q, V] -> [N, Q*V]
        #[filter_height, filter_width, in_channels, out_channels]
        """
        pooled = []
        for filter_size, num_filter in zip(filter_sizes, num_filters):
            with tf.variable_scope("conv-max_%s" % filter_size):
                
                weight_initializer = variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=False)
                bias_initializer = variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
                conv = tf.contrib.layers.convolution2d(inputs = D_q,
                                                num_outputs = num_filter,
                                                kernel_size = [filter_size, V],
                                                stride = [1, 1],
                                                padding = 'VALID',
                                                activation_fn = tf.nn.relu,
                                                weights_initializer = weight_initializer,
                                                biases_initializer = bias_initializer) # [N, Q-k+1, 1, num_filter]
                pool = tf.contrib.layers.max_pool2d(inputs = conv,
                                                    kernel_size = [Q - filter_size + 1, 1],
                                                    stride = [1, 1],
                                                    padding = 'valid') #[N, 1, 1, num_filter]
                pooled.append(pool)
        D_qf = tf.reshape(tf.concat(pooled, 3), [-1, sum(num_filters)]) #[N, num_filters]
        """
        with tf.variable_scope("fc_layers"):
            weight_initializer = variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=False)
            bias_initializer = variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
            stack_args = [[32, tf.nn.relu],
                          [2, None]]
            D_logits = tf.contrib.layers.stack(D_qf,
                                               fully_connected,
                                               stack_args,
                                               weights_initializer=weight_initializer,
                                               biases_initializer=bias_initializer,
                                               scope="fc_layers")
        return D_logits
            


    def QA_branch(self, embedding, qa_q, qa_story):
        params = self.params
        # attention mechanism
        #q_cell = rnn.GRUCell(params.seq2seq_hidden_size)
        q_cell = rnn.LSTMCell(params.seq2seq_hidden_size)
        num_layers = 1
        #q_cell = tf.contrib.rnn.MultiRNNCell([rnn.LSTMCell(params.seq2seq_hidden_size) for l in range(num_layers)])
        go_pad = tf.ones(tf.stack([self.batch_size, 1]), dtype=tf.int32)
        go_pad = tf.nn.embedding_lookup(embedding, go_pad) # [N, 1, V]
        decoder_inputs = tf.unstack(go_pad, axis=1) # 1 * [N, V]

        initial_state = (qa_q, qa_q)
        #initial_state = [(qa_q, qa_q) for l in range(num_layers)]
        attention_states = qa_story
        q_logprobs, _ = tf.contrib.legacy_seq2seq.attention_decoder(decoder_inputs=decoder_inputs,
                                                                    initial_state=initial_state,
                                                                    attention_states=attention_states,
                                                                    cell=q_cell,
                                                                    output_size=self.words.vocab_size,
                                                                    loop_function=None)
        q_logprobs[0] = tf.contrib.layers.batch_norm(q_logprobs[0], decay=0.9, is_training=self.is_training, center=True, scale=True,
                                                     updates_collections=None, scope='BatchNorm')

        return q_logprobs[0]

    def QG_branch(self, embedding, qg_q, qg_story, feed_previous, is_training):
        params = self.params
        L, Q, F = params.sentence_size, params.question_size, params.story_size
        V, A = params.seq2seq_hidden_size, self.words.vocab_size
        ## output projection weight ##
        """
        proj_w = weight('proj_w', [parmas.dmn_embedding_size, self.words.vocab_size])
        proj_b = bias('proj_b', self.words.vocab_size)
        """
        ## build decoder inputs ##
        go_pad = tf.ones(tf.stack([self.batch_size, 1]), dtype=tf.int32)
        go_pad = tf.nn.embedding_lookup(embedding, go_pad) # [N, 1, V]
        decoder_inputs = tf.concat(axis=1, values=[go_pad, qg_q]) # [N, Q+1, V]
        decoder_inputs = tf.unstack(decoder_inputs, axis=1)[:-1] # Q * [N, V]

        ## output idxs ##
        chosen_idxs = []

        ## question module rnn cell ##
        #q_cell = rnn.GRUCell(params.seq2seq_hidden_size)
        q_cell = rnn.LSTMCell(params.seq2seq_hidden_size)
        num_layers = 1
        #q_cell = tf.contrib.rnn.MultiRNNCell([rnn.LSTMCell(params.seq2seq_hidden_size) for l in range(num_layers)])

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
        tf.summary.scalar("explore_eps", explore_eps, collections=["VAR_SUMM"])
        ## decoder loop function ##
        def _loop_fn(prev, i):
            # prev = tf.matmul(prev, proj_w) + proj_b
            prev_symbol = tf.cond(self.is_sample,#tf.logical_and(is_training, feed_previous),
                                  lambda: tf.argmax(prev, 1),#gumbel_softmax(prev / explore_eps, 1),
                                  lambda: tf.argmax(prev, 1))
            chosen_idxs.append(prev_symbol)
            emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
            next_inp = tf.cond(feed_previous,
                               lambda: emb_prev,
                               lambda: decoder_inputs[i])
            return next_inp
        ## decoder ##
        def decoder(feed_previous_bool):
            loop_function = _loop_fn if feed_previous_bool else None
            reuse = None if feed_previous_bool else True
            with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                initial_state = (qg_story[:, -1], qg_story[:, -1])
                #initial_state = [(qg_story[:, -1], qg_story[:, -1]) for l in range(num_layers)]
                attention_states = qg_story
                q_logprobs, _ = tf.contrib.legacy_seq2seq.attention_decoder(decoder_inputs=decoder_inputs,
                                                                            initial_state=initial_state,
                                                                            attention_states=attention_states,
                                                                            cell=q_cell,
                                                                            output_size=self.words.vocab_size,
                                                                            loop_function=loop_function)
                q_logprobs = tf.contrib.layers.batch_norm(q_logprobs, decay=0.9, is_training=self.is_training, center=True, scale=True,
                                                          updates_collections=None, scope='BatchNorm')
                return q_logprobs
        """
        q_logprobs = tf.cond(feed_previous,
                             lambda: decoder(True),
                             lambda: decoder(False))
        """
        q_logprobs = decoder(True)

        last_symbol = tf.cond(self.is_sample,#tf.logical_and(is_training, feed_previous),
                              lambda: tf.argmax(q_logprobs[-1], 1),#gumbel_softmax(q_logprobs[-1] / explore_eps, 1),
                              lambda: tf.argmax(q_logprobs[-1], 1))
        chosen_idxs.append(last_symbol)
        assert len(chosen_idxs) == Q
        return q_logprobs, chosen_idxs
        # q_logprobs = [tf.matmul(out, proj_w) + proj_b for out in q_outputs]
        # return q_outputs

    def def_run_list(self):
        self.pre_train_list = [self.merged_QA, self.merged_QG, self.Pre_global_step, self.Pre_opt_op]
        self.QA_train_list  = [self.merged_QA, self.QA_global_step, self.QA_opt_op]
        self.rl_train_list  = [self.merged_RL, self.RL_global_step, self.RL_opt_op]
        #self.rl_train_list  = [self.merged_RL, self.RL_global_step, self.RL_global_step]
        self.D_train_list   = [self.merged_D, self.D_global_step, self.D_opt_op]
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

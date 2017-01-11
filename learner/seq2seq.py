import os
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.python.ops import rnn_cell
#from tensorflow.contrib.rnn import static_bidirectional_rnn
from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn

from learner.base_model import BaseModel
from dmn_helper.nn import weight, bias, dropout, batch_norm, variable_summary, gumbel_softmax
from ren_helper.model_utils import get_sequence_length


EPS = 1e-20


class Seq2Seq(BaseModel):
    def build(self, forward_only):
        params = self.params
        N, L, Q, F = params.batch_size, params.sentence_size, params.question_size, params.story_size
        V, d, A = params.dmn_embedding_size, params.dmn_embedding_size, self.words.vocab_size

        # initialize self
        # placeholders
        input = tf.placeholder('int32', shape=[N, F, L], name='x')  # [num_batch, fact_count, sentence_len]
        question = tf.placeholder('int32', shape=[N, Q], name='q')  # [num_batch, question_len]
        answer = tf.placeholder('int32', shape=[N], name='y')  # [num_batch] - one word answer
        fact_counts = get_sequence_length(input)
        is_training = tf.placeholder(tf.bool)
        feed_previous = tf.placeholder(tf.bool)

        # Prepare parameters
        gru = rnn_cell.GRUCell(d)
        l = self.positional_encoding()
        with tf.variable_scope('Embedding'):
            embedding = weight('embedding', [A, V], init='uniform', range=3**(1/2))
            embedding_mask = tf.constant([0 if i == 0 else 1 for i in range(A)], dtype=tf.float32, shape=[A, 1])
            embedding = embedding * embedding_mask
            variable_summary([embedding])

        with tf.name_scope('Tied_SentenceReader'):
            input_embed = tf.nn.embedding_lookup(embedding, input) # [N, F, L] -> [N, F, L, V]
            # apply positional encoding
            encoded = l * input_embed
            facts = tf.reduce_sum(encoded, 2)  # [N, F, V]
            # dropout
            facts = dropout(facts, params.dmn_keep_prob, is_training)

        with tf.name_scope('Tied_InputFusion') as scope:
            # Bidirectional RNN
            """
            f_outputs, fw_states, bw_states = stack_bidirectional_dynamic_rnn([gru],
                                                                              [gru],
                                                                              facts,
                                                                              sequence_length=fact_counts,
                                                                              dtype=tf.float32)
            # facts: [N, F, d*2]
            gru_variables = [v for v in tf.trainable_variables() if v.name.startswith(scope)]
            variable_summary(gru_variables)
            f_outputs = tf.split(2, 2, f_outputs) # [N, F, d*2] -> 2*[N, F, d]
            f_outputs = f_outputs[0] + f_outputs[1]
            """
            with tf.variable_scope('Forward') as scope:
                fw_output, fw_state = tf.nn.dynamic_rnn(gru, facts, fact_counts, dtype=tf.float32)
                gru_variables = [v for v in tf.trainable_variables() if v.name.startswith(scope.name)]
                variable_summary(gru_variables)
            with tf.variable_scope('Backward') as scope:
                facts_reverse = tf.reverse_sequence(facts, fact_counts, 1)
                bw_output, bw_state = tf.nn.dynamic_rnn(gru, facts_reverse, fact_counts, dtype=tf.float32)
                bw_output_rev = tf.reverse_sequence(bw_output, fact_counts, 1)
                gru_variables = [v for v in tf.trainable_variables() if v.name.startswith(scope.name)]
                variable_summary(gru_variables)
            # Use forward and backward states both for QA branch
            f_outputs = fw_output + bw_output_rev # [N, F, D]
            # QG branch init state [N, d]
            #q_init_state = fw[0] + bw[0]
            q_init_state = fw_state + bw_output[:, 0, :] # [N, d]

        with tf.variable_scope('Question_embed'):
            ques_embed = tf.nn.embedding_lookup(embedding, question) # [N, Q, V]

        ## QA Branch
        with tf.variable_scope('QA_branch'):
            QA_ans_logits = self.QA_branch(gru, ques_embed, f_outputs, is_training)
            QA_ans = tf.nn.softmax(QA_ans_logits)

            with tf.name_scope('QA_Loss'):
                # Cross-Entropy loss
                QA_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(QA_ans_logits, answer)
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
                QG_loss = tf.nn.seq2seq.sequence_loss(q_logprobs, target_list,
                                                   [tf.constant(np.ones((N,)), dtype=tf.float32)] * Q )
                QG_total_loss = QG_loss + params.dmn_weight_decay * tf.add_n(tf.get_collection('l2'))

        # Policy Gradient
        chosen_one_hot = tf.placeholder(tf.float32, shape=[N, Q, A], name='act')
        rewards = tf.placeholder(tf.float32, shape=[N], name='rewards')

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
        if forward_only:
            self.opt_op = None
        else:
            self.RL_OptOP = self.RLOpt()
            self.PreQA_OptOP = self.PreTrainQAOpt()
            self.PreQG_OptOP = self.PreTrainQGOpt()

    def QA_branch(self, gru, ques_embed, f_outputs, is_training):
        params = self.params
        N, L, Q, F = params.batch_size, params.sentence_size, params.question_size, params.story_size
        V, d, A = params.dmn_embedding_size, params.dmn_embedding_size, self.words.vocab_size
        # question vector
        with tf.variable_scope('question_vec'):
            ques_embed_list = tf.unstack(ques_embed, axis=1) # Q * [N, V]
            _, question_vec = tf.nn.rnn(gru, ques_embed_list, dtype=tf.float32) # [N, d]
        # Episodic Memory
        with tf.variable_scope('Episodic'):
            episode = EpisodeModule(d, question_vec, f_outputs, is_training, params.dmn_batch_norm)
            memory = tf.identity(question_vec) # [N, d]

            for t in range(params.dmn_memory_step):
                with tf.variable_scope('Layer%d' % t) as scope:
                    if params.dmn_memory_update == 'gru':
                        memory = gru(episode.new(memory), memory)[0]
                    else:
                        # ReLU update
                        c = episode.new(memory)
                        concated = tf.concat(1, [memory, c, question_vec])

                        w_t = weight('w_t', [3 * d, d])
                        z = tf.matmul(concated, w_t)
                        if params.dmn_batch_norm:
                            z = batch_norm(z, is_training)
                        else:
                            b_t = bias('b_t', d)
                            z = z + b_t
                        memory = tf.nn.relu(z)  # [N, d]

        # Regularizations
        if params.dmn_batch_norm:
            memory = batch_norm(memory, is_training=is_training)
        memory = dropout(memory, params.dmn_keep_prob, is_training)

        with tf.name_scope('QA_Answer'):
            # Answer module : feed-forward version (for it is one word answer)
            w_a = weight('w_a', [d, A], init='xavier')
            QA_ans_logits = tf.matmul(memory, w_a)  # [N, A]

        return QA_ans_logits

    def QG_branch(self, embedding, ques_embed, q_init_state, feed_previous):
        params = self.params
        N, L, Q, F = params.batch_size, params.sentence_size, params.question_size, params.story_size
        V, d, A = params.dmn_embedding_size, params.dmn_embedding_size, self.words.vocab_size
        ## output projection weight ##
        proj_w = weight('proj_w', [d, A])
        proj_b = bias('proj_b', A)
        ## build decoder inputs ##
        go_pad = tf.constant(np.ones((N, 1)), dtype=tf.int32)
        go_pad = tf.nn.embedding_lookup(embedding, go_pad) # [N, 1, V]
        decoder_inputs = tf.concat(1, [go_pad, ques_embed]) # [N, Q+1, V]
        decoder_inputs = tf.unstack(decoder_inputs, axis=1)[:-1] # Q * [N, V]
        ## question module rnn cell ##
        q_cell = rnn_cell.GRUCell(d)
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
                q_outputs, _ = tf.nn.seq2seq.rnn_decoder(decoder_inputs=decoder_inputs,
                                                         initial_state=q_init_state,
                                                         cell=q_cell,
                                                         loop_function=loop_function)
                return q_outputs
        q_outputs = tf.cond(feed_previous,
                            lambda: decoder(True),
                            lambda: decoder(False))
        q_logprobs = [tf.matmul(out, proj_w) + proj_b for out in q_outputs]

        return q_logprobs

    def PreTrainQAOpt(self):
        with tf.name_scope("PreTrainOpt"):
            def learning_rate_decay_fn(lr, global_step):
                return tf.train.exponential_decay(lr,
                                                  global_step,
                                                  decay_steps=5000,
                                                  decay_rate=0.95,
                                                  staircase=True)
            OPTIMIZER_SUMMARIES = ["learning_rate",
                                   "loss",
                                   "gradients",
                                   "gradient_norm"] if self.action == 'train' else []
            opt_op = tf.contrib.layers.optimize_loss(self.QA_total_loss,
                                                     self.global_step,
                                                     learning_rate=self.params.learning_rate,
                                                     optimizer=tf.train.AdamOptimizer,
                                                     clip_gradients=5.,
                                                     learning_rate_decay_fn=learning_rate_decay_fn,
                                                     summaries=OPTIMIZER_SUMMARIES)
        return opt_op

    def PreTrainQGOpt(self):
        with tf.name_scope("PreTrainOpt"):
            def learning_rate_decay_fn(lr, global_step):
                return tf.train.exponential_decay(lr,
                                                  global_step,
                                                  decay_steps=5000,
                                                  decay_rate=0.95,
                                                  staircase=True)
            OPTIMIZER_SUMMARIES = ["learning_rate",
                                   "loss",
                                   "gradients",
                                   "gradient_norm"] if self.action == 'train' else []
            opt_op = tf.contrib.layers.optimize_loss(self.QG_total_loss,
                                                     self.global_step,
                                                     learning_rate=self.params.learning_rate,
                                                     optimizer=tf.train.AdamOptimizer,
                                                     clip_gradients=5.,
                                                     learning_rate_decay_fn=learning_rate_decay_fn,
                                                     summaries=OPTIMIZER_SUMMARIES)
        return opt_op

    def RLOpt(self):
        with tf.name_scope("RLOpt"):
            def learning_rate_decay_fn(lr, global_step):
                return tf.train.exponential_decay(lr,
                                                  global_step,
                                                  decay_steps=5000,
                                                  decay_rate=0.95,
                                                  staircase=True)
            OPTIMIZER_SUMMARIES = ["learning_rate",
                                   "loss",
                                   "gradients",
                                   "gradient_norm"] if self.action == 'rl' else []
            opt_op = tf.contrib.layers.optimize_loss(self.J,
                                                     self.global_step,
                                                     learning_rate=self.params.learning_rate,
                                                     optimizer=tf.train.AdamOptimizer,
                                                     clip_gradients=5.,
                                                     learning_rate_decay_fn=learning_rate_decay_fn,
                                                     summaries=OPTIMIZER_SUMMARIES)
        return opt_op

    def positional_encoding(self):
        V, L = self.params.dmn_embedding_size, self.params.sentence_size
        encoding = np.zeros([L, V])
        for l in range(L):
            for v in range(V):
                encoding[l, v] = (1 - float(l)/L) - (float(v)/V)*(1 - 2.0*l/L)

        return encoding

    def get_feed_dict(self, batches, feed_previous, is_training):
        return {
            self.x: batches[0],
            self.q: batches[1],
            self.y: batches[2],
            self.is_training: is_training,
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

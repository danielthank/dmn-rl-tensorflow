import os
import json
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.contrib import rnn

from learner.base_model import BaseModel
from dmn_helper.episode_module import EpisodeModule
from dmn_helper.nn import weight, bias, dropout, batch_norm, variable_summary, gumbel_softmax
from ren_helper.model_utils import get_sequence_length


EPS = 1e-20


class DMN(BaseModel):
    """ Dynamic Memory Networks (March 2016 Version - https://arxiv.org/abs/1603.01417)
        Improved End-To-End version."""
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
        # input_mask = tf.placeholder('float32', shape=[N, F, L, V], name='xm')
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

        with tf.name_scope('SentenceReader'):
            input_embed = tf.nn.embedding_lookup(embedding, input) # [N, F, L] -> [N, F, L, V]
            # apply positional encoding
            #encoded = l * input_embed * input_mask
            encoded = l * input_embed
            facts = tf.reduce_sum(encoded, 2)  # [N, F, V]
            # dropout time
            facts = dropout(facts, params.dmn_keep_prob, is_training)

        with tf.name_scope('InputFusion'):
            # Bidirectional RNN
            with tf.variable_scope('Forward') as scope:
                forward_states, _ = tf.nn.dynamic_rnn(gru, facts, fact_counts, dtype=tf.float32)
                gru_variables = [v for v in tf.trainable_variables() if v.name.startswith(scope.name)]
                variable_summary(gru_variables)

            with tf.variable_scope('Backward') as scope:
                facts_reverse = tf.reverse_sequence(facts, fact_counts, 1)
                backward_states, _ = tf.nn.dynamic_rnn(gru, facts_reverse, fact_counts, dtype=tf.float32)
                backward_states = tf.reverse_sequence(backward_states, fact_counts, 1)
                gru_variables = [v for v in tf.trainable_variables() if v.name.startswith(scope.name)]
                variable_summary(gru_variables)

            # Use forward and backward states both
            facts = forward_states + backward_states  # [N, F, d]

        with tf.name_scope('Answer'):
            answer_vec = tf.nn.embedding_lookup(embedding, answer)

        # Episodic Memory
        with tf.variable_scope('Episodic') as scope:
            episode = EpisodeModule(d, answer_vec, facts, is_training, params.dmn_batch_norm)
            memory = tf.identity(answer_vec) # [N, d]

            for t in range(params.dmn_memory_step):
                with tf.variable_scope('Layer%d' % t):
                    if params.dmn_memory_update == 'gru':
                        memory = gru(episode.new(memory), memory)[0]
                    else:
                        # ReLU update
                        c = episode.new(memory)
                        concated = tf.concat(axis=1, values=[memory, c, answer_vec])

                        w_t = weight('w_t', [3 * d, d])
                        z = tf.matmul(concated, w_t)
                        if params.dmn_batch_norm:
                            z = batch_norm(z, is_training)
                        else:
                            b_t = bias('b_t', d)
                            z = z + b_t
                        memory = tf.nn.relu(z)  # [N, d]
                #scope.reuse_variables()

            variables = [v for v in tf.trainable_variables() if v.name.startswith(scope.name)]
            variable_summary(variables)

            # variable_summary([episode.w1, episode.b1, episode.w2, episode.b2])
            if params.dmn_batch_norm:
                memory = batch_norm(memory, is_training=is_training)
            memory = dropout(memory, params.dmn_keep_prob, is_training)

        with tf.variable_scope('Question') as scope:
            ## output projection weight ##
            proj_w = weight('proj_w', [d, A])
            proj_b = bias('proj_b', A)
            ## build decoder inputs ##
            go_pad = tf.constant(np.ones((N, 1)), dtype=tf.int32)
            decoder_inputs = tf.concat(axis=1, values=[go_pad, question])
            decoder_inputs = tf.nn.embedding_lookup(embedding, decoder_inputs) # [N, Q+1, V]
            decoder_inputs = tf.transpose(decoder_inputs, [1, 0, 2]) # [Q+1, N, V]
            decoder_inputs = tf.unstack(decoder_inputs)[:-1] # Q * [N, V]
            decoder_inputs = [tf.concat(axis=1, values=[de_inp, memory]) for de_inp in decoder_inputs]
            ## question module rnn cell ##
            q_cell = rnn.GRUCell(d)
            ## decoder state init ##
            q_init_state = memory
            ## decoder loop function ##
            def _loop_fn(prev, i):
                prev = tf.matmul(prev, proj_w) + proj_b
                prev_symbol = tf.argmax(prev, 1)
                #prev_symbol = gumbel_softmax(prev, axis=1)
                emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
                return tf.concat(axis=1, values=[emb_prev, memory])
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
            q_probs = [tf.nn.softmax(out) for out in q_logprobs]
            variables = [v for v in tf.trainable_variables() if v.name.startswith(scope.name)]
            variable_summary(variables)

        with tf.name_scope('Loss'):
            target_list = tf.unstack(tf.transpose(question)) # Q * [N]
            # Cross-Entropy loss
            loss = tf.nn.seq2seq.sequence_loss(q_logprobs, target_list,
                                               [tf.constant(np.ones((N,)), dtype=tf.float32)] * Q )
            total_loss = loss + params.dmn_weight_decay * tf.add_n(tf.get_collection('l2'))

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

        self.chosen_one_hot = chosen_one_hot
        self.rewards = rewards


        # tensors
        self.total_loss = total_loss
        self.J = J
        self.q_probs = q_probs
        if not forward_only:
            self.RL_opt_op = self.RLOpt()
            self.Pre_opt_op = self.PreTrainOpt()

    def PreTrainOpt(self):
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
            opt_op = tf.contrib.layers.optimize_loss(self.total_loss,
                                                     self.global_step,
                                                     learning_rate=self.params.learning_rate,
                                                     optimizer=tf.train.AdamOptimizer,
                                                     clip_gradients=5.,
                                                     learning_rate_decay_fn=learning_rate_decay_fn,
                                                     summaries=OPTIMIZER_SUMMARIES)
            """
            optimizer = tf.train.AdamOptimizer(self.params.learning_rate)
            opt_op = optimizer.minimize(self.total_loss, global_step=self.global_step)
            variable_summary([lr])
            """
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
            """
            optimizer = tf.train.AdamOptimizer(self.params.learning_rate)
            opt_op = optimizer.minimize(self.J, global_step=self.global_step)
            variable_summary([lr])
            """
        return opt_op

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

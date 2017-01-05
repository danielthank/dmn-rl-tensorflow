import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.python.ops import rnn_cell

from GQ_base_model import GQBaseModel
from episode_module import EpisodeModule
from nn import weight, bias, dropout, batch_norm, variable_summary


class DMN(GQBaseModel):
    """ Dynamic Memory Networks (March 2016 Version - https://arxiv.org/abs/1603.01417)
        Improved End-To-End version."""
    def build(self, feed_previous, forward_only):
        params = self.params
        N, L, Q, F = params.batch_size, params.max_sent_size, params.max_ques_size, params.max_fact_count
        V, d, A = params.embed_size, params.hidden_size, self.words.vocab_size

        # initialize self
        # placeholders
        input = tf.placeholder('int32', shape=[N, F, L], name='x')  # [num_batch, fact_count, sentence_len]
        question = tf.placeholder('int32', shape=[N, Q], name='q')  # [num_batch, question_len]
        answer = tf.placeholder('int32', shape=[N], name='y')  # [num_batch] - one word answer
        fact_counts = tf.placeholder('int64', shape=[N], name='fc')
        input_mask = tf.placeholder('float32', shape=[N, F, L, V], name='xm')
        is_training = tf.placeholder(tf.bool)

        # Prepare parameters
        gru = rnn_cell.GRUCell(d)
        l = self.positional_encoding()
        with tf.variable_scope('Embedding'):
            embedding = weight('embedding', [A, V], init='uniform', range=3**(1/2))
            variable_summary([embedding])

        with tf.name_scope('SentenceReader'):
            input_list = tf.unpack(tf.transpose(input))  # L x [F, N]
            input_embed = []
            for facts in input_list:
                facts = tf.unpack(facts)
                embed = tf.pack([tf.nn.embedding_lookup(embedding, w) for w in facts])  # [F, N, V]
                input_embed.append(embed)
            # apply positional encoding
            input_embed = tf.transpose(tf.pack(input_embed), [2, 1, 0, 3])  # [L, F, N, V] -> [N, F, L, V]
            encoded = l * input_embed * input_mask
            facts = tf.reduce_sum(encoded, 2)  # [N, F, V]
            # dropout
            facts = dropout(facts, params.keep_prob, is_training)

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
            episode = EpisodeModule(d, answer_vec, facts, is_training, params.batch_norm)
            memory = tf.identity(answer_vec) # [N, d]

            for t in range(params.memory_step):
                with tf.variable_scope('Layer%d' % t):
                    if params.memory_update == 'gru':
                        memory = gru(episode.new(memory), memory)[0]
                    else:
                        # ReLU update
                        c = episode.new(memory)
                        concated = tf.concat(1, [memory, c, answer_vec])

                        w_t = weight('w_t', [3 * d, d])
                        z = tf.matmul(concated, w_t)
                        if params.batch_norm:
                            z = batch_norm(z, is_training)
                        else:
                            b_t = bias('b_t', d)
                            z = z + b_t
                        memory = tf.nn.relu(z)  # [N, d]
                #scope.reuse_variables()

            variables = [v for v in tf.trainable_variables() if v.name.startswith(scope.name)]
            variable_summary(variables)

            # variable_summary([episode.w1, episode.b1, episode.w2, episode.b2])
            if params.batch_norm:
                memory = batch_norm(memory, is_training=is_training)
            memory = dropout(memory, params.keep_prob, is_training)

        with tf.variable_scope('Question') as scope:
            proj_w = weight('proj_w', [d, A])
            proj_b = bias('proj_b', A)
            go_pad = tf.constant(np.ones((N, 1)), dtype=tf.int32)
            decoder_inputs = tf.concat(1, [go_pad, question])
            decoder_inputs = tf.nn.embedding_lookup(embedding, decoder_inputs) # [N, Q+1, V]
            decoder_inputs = tf.transpose(decoder_inputs, [1, 0, 2]) # [Q+1, N, V]
            decoder_inputs = tf.unstack(decoder_inputs)[:-1] # Q * [N, V]
            decoder_inputs = [tf.concat(1, [de_inp, memory]) for de_inp in decoder_inputs]
            q_cell = rnn_cell.GRUCell(d);
            q_init_state = memory
            if feed_previous:
                def _loop_fn(prev, i):
                    prev = tf.matmul(prev, proj_w) + proj_b
                    prev_symbol = tf.argmax(prev, 1)
                    emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol);
                    return tf.concat(1, [emb_prev, memory])
                loop_function = _loop_fn
            else:
                loop_function = None
            q_outputs, _ = tf.nn.seq2seq.rnn_decoder(decoder_inputs=decoder_inputs,
                                                     initial_state=q_init_state,
                                                     cell=q_cell,
                                                     loop_function=loop_function)
            q_outputs = [tf.matmul(out, proj_w) + proj_b for out in q_outputs]
            variables = [v for v in tf.trainable_variables() if v.name.startswith(scope.name)]
            variable_summary(variables)

        with tf.name_scope('Loss'):
            target_list = tf.unpack(tf.transpose(question)) # Q * [N]
            # Cross-Entropy loss
            loss = tf.nn.seq2seq.sequence_loss(q_outputs, target_list,
                                               [tf.constant(np.ones((N,)), dtype=tf.float32)] * Q )
            total_loss = loss + params.weight_decay * tf.add_n(tf.get_collection('l2'))

        # Training
        if not forward_only:
            def learning_rate_decay_fn(lr, global_step):
                return tf.train.exponential_decay(lr,
                                                  global_step,
                                                  decay_steps=5000,
                                                  decay_rate=0.95,
                                                  staircase=True)
            opt_op = tf.contrib.layers.optimize_loss(total_loss,
                                                     self.global_step,
                                                     learning_rate=params.learning_rate,
                                                     optimizer=tf.train.AdamOptimizer,
                                                     clip_gradients=5.,
                                                     learning_rate_decay_fn=learning_rate_decay_fn)
            """
            optimizer = tf.train.AdamOptimizer(params.learning_rate)
            opt_op = optimizer.minimize(total_loss, global_step=self.global_step)
            variable_summary([lr])
            """

        # placeholders
        self.x = input
        self.xm = input_mask
        self.q = question
        self.y = answer
        self.fc = fact_counts
        self.is_training = is_training

        # tensors
        self.total_loss = total_loss
        self.output = q_outputs
        if not forward_only:
            self.opt_op = opt_op

    def positional_encoding(self):
        V, L = self.params.embed_size, self.params.max_sent_size
        encoding = np.zeros([L, V])
        for l in range(L):
            for v in range(V):
                encoding[l, v] = (1 - float(l)/L) - (float(v)/V)*(1 - 2.0*l/L)

        return encoding

    def preprocess_batch(self, batches):
        """ Make padding and masks last word of sentence. (EOS token)
        :param batches: A tuple (input, question, label, mask)
        :return A tuple (input, question, label, mask)
        """
        params = self.params
        input, question, label = batches
        N, L, Q, F = params.batch_size, params.max_sent_size, params.max_ques_size, params.max_fact_count
        V = params.embed_size

        # make input and question fixed size
        new_input = np.zeros([N, F, L])  # zero padding
        input_masks = np.zeros([N, F, L, V])
        new_question = np.zeros([N, Q])
        new_labels = []
        fact_counts = []

        for n in range(N):
            for i, sentence in enumerate(input[n]):
                sentence_len = len(sentence)
                new_input[n, i, :sentence_len] = [self.words.word_to_index(w) for w in sentence]
                input_masks[n, i, :sentence_len, :] = 1.  # mask words

            fact_counts.append(len(input[n]))

            sentence_len = len(question[n])
            new_question[n, :sentence_len] = [self.words.word_to_index(w) for w in question[n]]

            new_labels.append(self.words.word_to_index(label[n]))

        return new_input, new_question, new_labels, fact_counts, input_masks

    def get_feed_dict(self, batches, is_train):
        input, question, label, fact_counts, mask = self.preprocess_batch(batches)
        return {
            self.x: input,
            self.xm: mask,
            self.q: question,
            self.y: label,
            self.fc: fact_counts,
            self.is_training: is_train
        }

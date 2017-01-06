import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.python.ops import rnn_cell
#from tensorflow.contrib.rnn import static_bidirectional_rnn
from tensorflow.contrib.rnn import stack_bidirectional_dynamic_rnn

from GQ_base_model import GQBaseModel
from episode_module import EpisodeModule
from nn import weight, bias, dropout, batch_norm, variable_summary, gumbel_softmax


class Seq2Seq(GQBaseModel):
    def build(self, forward_only):
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
        feed_previous = tf.placeholder(tf.bool)

        # Prepare parameters
        gru = rnn_cell.GRUCell(d)
        l = self.positional_encoding()
        with tf.variable_scope('Embedding'):
            embedding = weight('embedding', [A, V], init='uniform', range=3**(1/2))
            variable_summary([embedding])

        with tf.name_scope('SentenceReader'):
            input_embed = tf.nn.embedding_lookup(embedding, input) # [N, F, L] -> [N, F, L, V]
            # apply positional encoding
            encoded = l * input_embed * input_mask
            facts = tf.reduce_sum(encoded, 2)  # [N, F, V]
            # dropout
            """
            facts = dropout(facts, params.keep_prob, is_training)
            """

        with tf.name_scope('InputFusion') as scope:
            # Bidirectional RNN
            """
            f_outputs, fw, bw = stack_bidirectional_dynamic_rnn([gru],
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
                _, forward_state = tf.nn.dynamic_rnn(gru, facts, fact_counts, dtype=tf.float32)
                gru_variables = [v for v in tf.trainable_variables() if v.name.startswith(scope.name)]
                variable_summary(gru_variables)
            """
            with tf.variable_scope('Backward') as scope:
                facts_reverse = tf.reverse_sequence(facts, fact_counts, 1)
                backward_states, _ = tf.nn.dynamic_rnn(gru, facts_reverse, fact_counts, dtype=tf.float32)
                gru_variables = [v for v in tf.trainable_variables() if v.name.startswith(scope.name)]
                variable_summary(gru_variables)
            """
            """
            # Use forward and backward states both
            #f_outputs = forward_states + backward_states  # [N, F, d]
            f_outputs = forward_state
            """

        """
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
        """

        with tf.variable_scope('Question') as scope:
            ## output projection weight ##
            proj_w = weight('proj_w', [d, A])
            proj_b = bias('proj_b', A)
            ## build decoder inputs ##
            go_pad = tf.constant(np.ones((N, 1)), dtype=tf.int32)
            decoder_inputs = tf.concat(1, [go_pad, question])
            decoder_inputs = tf.nn.embedding_lookup(embedding, decoder_inputs) # [N, Q+1, V]
            decoder_inputs = tf.transpose(decoder_inputs, [1, 0, 2]) # [Q+1, N, V]
            decoder_inputs = tf.unstack(decoder_inputs)[:-1] # Q * [N, V]
            ## question module rnn cell ##
            q_cell = rnn_cell.GRUCell(d)
            ## decoder state init ##
            #q_init_state = q_cell.zero_state(batch_size=N, dtype=tf.float32)
            #q_init_state = fw[0] + bw[0]
            q_init_state = forward_state
            ## decoder loop function ##
            def _loop_fn(prev, i):
                prev = tf.matmul(prev, proj_w) + proj_b
                #prev_symbol = tf.argmax(prev, 1)
                prev_symbol = gumbel_softmax(prev, axis=1)
                now_input = tf.nn.embedding_lookup(embedding, prev_symbol)
                return now_input
            ## decoder ##
            def decoder(feed_previous_bool):
                loop_function = _loop_fn if feed_previous_bool else None
                reuse = None if feed_previous_bool else True
                with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                    q_outputs, _ = tf.nn.seq2seq.rnn_decoder(decoder_inputs=decoder_inputs,
                                                             initial_state=q_init_state,
                                                             cell=q_cell,
                                                             loop_function=loop_function)
                    """
                    q_outputs, _ = tf.nn.seq2seq.attention_decoder(decoder_inputs=decoder_inputs,
                                                                   initial_state=q_init_state,
                                                                   attention_states=f_outputs,
                                                                   cell=q_cell,
                                                                   loop_function=loop_function)
                    """
                    return q_outputs
            q_outputs = tf.cond(feed_previous,
                                lambda: decoder(True),
                                lambda: decoder(False))
            q_outputs = [tf.matmul(out, proj_w) + proj_b for out in q_outputs]
            variables = [v for v in tf.trainable_variables() if v.name.startswith(scope.name)]
            variable_summary(variables)

        with tf.name_scope('Loss'):
            target_list = tf.unstack(tf.transpose(question)) # Q * [N]
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
                                                  decay_rate=0.5,
                                                  staircase=True)
            OPTIMIZER_SUMMARIES = ["learning_rate",
                                   "loss",
                                   "gradients",
                                   "gradient_norm"]
            opt_op = tf.contrib.layers.optimize_loss(total_loss,
                                                     self.global_step,
                                                     learning_rate=params.learning_rate,
                                                     optimizer=tf.train.AdamOptimizer,
                                                     clip_gradients=5.,
                                                     learning_rate_decay_fn=learning_rate_decay_fn,
                                                     summaries=OPTIMIZER_SUMMARIES)
            """
            optimizer = tf.train.AdamOptimizer(params.learning_rate)
            opt_op = optimizer.minimize(total_loss, global_step=self.global_step)
            """

        # placeholders
        self.x = input
        self.xm = input_mask
        self.q = question
        self.y = answer
        self.fc = fact_counts
        self.is_training = is_training
        self.feed_previous = feed_previous

        # tensors
        self.total_loss = total_loss
        self.output = q_outputs
        if not forward_only:
            self.opt_op = opt_op
        else:
            self.opt_op = None

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

    def get_feed_dict(self, batches, feed_previous, is_train):
        input, question, label, fact_counts, mask = self.preprocess_batch(batches)
        return {
            self.x: input,
            self.xm: mask,
            self.q: question,
            self.y: label,
            self.fc: fact_counts,
            self.is_training: is_train,
            self.feed_previous: feed_previous
        }

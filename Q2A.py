import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.python.ops import rnn_cell

from base_model import BaseModel
from episode_module import EpisodeModule
from nn import weight, bias, dropout, batch_norm, variable_summary


class DMN(BaseModel):
    """ Dynamic Memory Networks (March 2016 Version - https://arxiv.org/abs/1603.01417)
        Improved End-To-End version."""
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

        # Prepare parameters
        gru = rnn_cell.GRUCell(d)
        l = self.positional_encoding()
        embedding = weight('embedding', [A, V], init='uniform', range=3**(1/2))

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

        # dropout time
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


            # Use forward and backward states both
            facts = forward_states + backward_states  # [N, F, d]

        with tf.variable_scope('Question'):
            ques_list = tf.unpack(tf.transpose(question)) # Q * [N]
            ques_embed = [tf.nn.embedding_lookup(embedding, w) for w in ques_list] # Q * [N, V]
            _, question_vec = tf.nn.rnn(gru, ques_embed, dtype=tf.float32) # [N, d]

        # Episodic Memory
        with tf.variable_scope('Episodic'):
            episode = EpisodeModule(d, question_vec, facts, is_training, params.batch_norm)
            memory = tf.identity(question_vec) # [N, d]

            for t in range(params.memory_step):
                with tf.variable_scope('Layer%d' % t) as scope:
                    if params.memory_update == 'gru':
                        memory = gru(episode.new(memory), memory)[0]
                    else:
                        # ReLU update
                        c = episode.new(memory)
                        concated = tf.concat(1, [memory, c, question_vec])

                        w_t = weight('w_t', [3 * d, d])
                        z = tf.matmul(concated, w_t)
                        if params.batch_norm:
                            z = batch_norm(z, is_training)
                        else:
                            b_t = bias('b_t', d)
                            z = z + b_t
                        memory = tf.nn.relu(z)  # [N, d]

                    scope.reuse_variables()

        # Regularizations
        if params.batch_norm:
            memory = batch_norm(memory, is_training=is_training)
        memory = dropout(memory, params.keep_prob, is_training)

        with tf.name_scope('Answer'):
            # Answer module : feed-forward version (for it is one word answer)
            w_a = weight('w_a', [d, A], init='xavier')
            logits = tf.matmul(memory, w_a)  # [N, A]

        with tf.name_scope('Loss'):
            # Cross-Entropy loss
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, answer)
            loss = tf.reduce_mean(cross_entropy)
            total_loss = loss + params.weight_decay * tf.add_n(tf.get_collection('l2'))

        with tf.variable_scope('Accuracy'):
            # Accuracy
            predicts = tf.cast(tf.argmax(logits, 1), 'int32')
            corrects = tf.equal(predicts, answer)
            num_corrects = tf.reduce_sum(tf.cast(corrects, tf.float32))
            accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

        # Training
        if not forward_only:
            def learning_rate_decay_fn(lr, global_step):
                return tf.train.exponential_decay(lr,
                                                  global_step,
                                                  decay_steps=3000,
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
        self.output = tf.nn.softmax(logits)
        self.total_loss = total_loss
        self.num_corrects = num_corrects
        self.accuracy = accuracy
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

import tensorflow as tf

from dmn_helper.nn import weight, bias, variable_summary
from dmn_helper.attn_gru import AttnGRU


class EpisodeModule:
    """ Inner GRU module in episodic memory that creates episode vector. """
    def __init__(self, num_hidden, question, facts, is_training, bn):
        self.question = question
        self.facts = tf.unpack(tf.transpose(facts, [1, 0, 2]))  # F x [N, d]
        self.num_hidden = num_hidden
        self.gru = AttnGRU(num_hidden, is_training, bn)
        with tf.variable_scope('w1b1w2b2'):
            self.w1 = weight('w1', [4 * self.num_hidden, self.num_hidden])
            self.b1 = bias('b1', [self.num_hidden])
            self.w2 = weight('w2', [self.num_hidden, 1])
            self.b2 = bias('b2', [1])

    @property
    def init_state(self):
        return tf.zeros_like(self.facts[0])

    def new(self, memory):
        """ Creates new episode vector (will feed into Episodic Memory GRU)
        :param memory: Previous memory vector
        :return: episode vector
        """
        with tf.variable_scope('gru'):
            state = self.init_state

            for f in self.facts:
                g = self.attention(f, memory)
                state = self.gru(f, state, g)
                tf.get_variable_scope().reuse_variables()

            # gru_variables = [v for v in tf.all_variables() if v.name.startswith(scope.name)]

        return state

    def attention(self, f, m):
        """ Attention mechanism. For details, see paper.
        :param f: A fact vector [N, d] at timestep
        :param m: Previous memory vector [N, d]
        :return: attention vector at timestep
        """
        with tf.name_scope('attention') as scope:
            # NOTE THAT instead of L1 norm we used L2
            q = self.question
            vec = tf.concat(1, [f * q, f * m, tf.abs(f - q), tf.abs(f - m)])  # [N, 4*d]
            # attention learning
            l1 = tf.matmul(vec, self.w1) + self.b1
            l1 = tf.nn.tanh(l1)
            l2 = tf.matmul(l1, self.w2) + self.b2 # [N, 1]
            return tf.nn.softmax(l2)

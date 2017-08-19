import os
import json
import numpy as np
import tensorflow as tf
import inspect

from functools import partial

from lm.base_model import BaseModel
from tf_helper.model_utils import get_sequence_length

class RNNLM(BaseModel):
    def build(self, eval_flag):
        params = self.params
        batch_size = params.batch_size
        vocab_size = self.vocab_size
        num_layers = params.rnnlm_layers
        hidden_size = params.rnnlm_hidden_size
        num_steps = self.num_steps

        # initialize self
        # placeholders
        inputs = tf.placeholder('int32', shape=[None, num_steps], name='x')  # [num_batch, num_steps]
        ground_truth = tf.placeholder('int32', shape=[None, num_steps], name='y')  # [num_batch, num_steps] - right shift version of x
        is_training = tf.placeholder(tf.bool)
        keep_prob = tf.placeholder(tf.float32)
        batch_size = tf.shape(inputs)[0]

        normal_initializer = tf.random_normal_initializer(stddev=0.1)
        ones_initializer = tf.constant_initializer(1.0)

        with tf.variable_scope('RNNLM', initializer=normal_initializer):
            # Embeddings
            with tf.device('/cpu:0'):
                embedding_params = tf.get_variable('embedding_params', [vocab_size, hidden_size])

                inputs_embedding = tf.nn.embedding_lookup(embedding_params, inputs)
            inputs_embedding = tf.cond(is_training, 
                                        lambda: tf.nn.dropout(inputs_embedding, keep_prob),
                                        lambda: inputs_embedding)

            # Recurrence

            # Slightly better results can be obtained with forget gate biases
            # initialized to 1 but the hyperparameters of the model would need to be
            # different than reported in the paper.
            def lstm_cell():
              # With the latest TensorFlow source code (as of Mar 27, 2017),
              # the BasicLSTMCell will need a reuse parameter which is unfortunately not
              # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
              # an argument check here:
                if 'reuse' in inspect.getargspec(
                    tf.contrib.rnn.BasicLSTMCell.__init__).args:
                    return tf.contrib.rnn.BasicLSTMCell(
                        hidden_size, forget_bias=0.0, state_is_tuple=True,
                        reuse=tf.get_variable_scope().reuse)
                else:
                    return tf.contrib.rnn.BasicLSTMCell(
                        hidden_size, forget_bias=0.0, state_is_tuple=True)
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=keep_prob)
            cell = tf.contrib.rnn.MultiRNNCell(
                [attn_cell() for _ in range(num_layers)], state_is_tuple=True)

            self.initial_state = cell.zero_state(batch_size, 'float32')
            
            ### RNN ###
            outputs = []
            state = self.initial_state
            with tf.variable_scope('core'):
                for time_step in range(num_steps):
                    if time_step > 0:tf.get_variable_scope().reuse_variables()
                    (cell_output, state) = cell(inputs_embedding[:, time_step, :], state)
                    outputs.append(cell_output)
            
            output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, hidden_size]) #[batch_size * num_steps, hidden_size]
            weight_o = tf.get_variable('output_weight', [hidden_size, vocab_size], 'float32')
            bias_o = tf.get_variable('output_bias', [vocab_size], 'float32')
            logits = tf.matmul(output, weight_o) + bias_o
            
            loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                    [logits], [tf.reshape(ground_truth, [-1])],
                    [tf.ones([batch_size * num_steps], dtype='float32')])
            batch_size = tf.cast(batch_size, tf.float32) 
            self.loss = tf.reduce_sum(loss) / batch_size
            log_perp = loss
            self.final_state = state # for recurrent purpose
            if not eval_flag:
                def learning_rate_decay_fn(lr, global_step):
                    return tf.train.exponential_decay(lr,
                                                      global_step,
                                                      decay_steps=3000,
                                                      decay_rate=0.8,
                                                      staircase=True)
                OPTIMIZER_SUMMARIES = ["learning_rate",
                                       "loss",
                                       "gradients",
                                       "gradient_norm"]
                opt_op = tf.contrib.layers.optimize_loss(self.loss,
                                                         self.global_step,
                                                         learning_rate=params.learning_rate,
                                                         optimizer=tf.train.AdamOptimizer,
                                                         clip_gradients=10.,
                                                         learning_rate_decay_fn=learning_rate_decay_fn,
                                                         summaries=OPTIMIZER_SUMMARIES)
            self.x = inputs
            self.y = ground_truth
            self.is_training = is_training
            self.keep_prob = keep_prob
            self.log_perp = log_perp #[batch_size, 1]
            self.num_steps = num_steps
            # Output Module
            if not eval_flag:
                self.opt_op = opt_op
            else:
                self.opt_op = None


    def get_feed_dict(self, batches, is_train, state):
        feed_dict = {
            self.x: batches[0],
            self.y: batches[1],
            self.is_training: is_train,
            self.keep_prob: self.params.rnnlm_keep_prob if is_train else 1.0
        }
        for i, (c, h) in enumerate(self.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        return feed_dict
    def save_params(self):
        assert self.action == 'train'
        params = self.params
        filename = os.path.join(self.save_dir, "params.json")
        save_params_dict = {'rnnlm_layers': params.rnnlm_layers,
                            'rnnlm_hidden_size': params.rnnlm_hidden_size,
                            'target': params.target,
                            'arch': params.arch,
                            'task': params.task}
        with open(filename, 'w') as file:
            json.dump(save_params_dict, file, indent=4)

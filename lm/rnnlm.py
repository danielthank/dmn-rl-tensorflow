import os
import json
import numpy as np
import tensorflow as tf
import inspect

from functools import partial

from lm.base_model import BaseModel
from tf_helper.model_utils import get_sequence_length
from ren_helper.dynamic_memory_cell import DynamicMemoryCell

class RNNLM(BaseModel):
    def build(self):
        params = self.params
        batch_size, sentence_size, question_size, story_size = params.batch_size, params.sentence_size, params.question_size, params.story_size
        num_steps = max(sentence_size, question_size, story_size)
        keep_prob = params.rnnlm_keep_prob
        embedding_size, vocab_size = params.ren_embedding_size, self.words.vocab_size
        num_layers = params.rnnlm_layers
        hidden_size = params.rnnlm_hidden_size 

        # initialize self
        # placeholders
        inputs = tf.placeholder('int32', shape=[None, num_steps], name='x')  # [num_batch, num_steps]
        ground_truth = tf.placeholder('int32', shape=[None, num_steps], name='y')  # [num_batch, num_steps] - right shift version of x
        is_training = tf.placeholder(tf.bool)
        batch_size = tf.shape(inputs)[0]

        normal_initializer = tf.random_normal_initializer(stddev=0.1)
        ones_initializer = tf.constant_initializer(1.0)

        with tf.variable_scope('RNNLM', initializer=normal_initializer):
            # Embeddings
            # The embedding mask forces the special "pad" embedding to zeros.
            embedding_params = tf.get_variable('embedding_params', [vocab_size, embedding_size])
            embedding_mask = tf.constant([0 if i == 0 else 1 for i in range(vocab_size)],
                dtype=tf.float32,
                shape=[vocab_size, 1])
            embedding_params_masked = embedding_params * embedding_mask

            inputs_embedding = tf.nn.embedding_lookup(embedding_params_masked, inputs)
            if is_training and keep_prob < 1:
                inputs_embedding = tf.nn.dropout(inputs_embedding, keep_prob)

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
            attn_cell = lstm_cell
            if is_training and keep_prob < 1:
                def attn_cell():
                    return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob=keep_prob)
            cell = tf.contrib.rnn.MultiRNNCell(
                [attn_cell() for _ in range(num_layers)], state_is_tuple=True)

            self._initial_state = cell.zero_state(batch_size, 'float32')
            
            ### RNN ###
            outputs = []
            state = self._initial_state
            with tf.variable_scope('core'):
                for time_step in range(num_steps):
                    if time_step > 0:
                        tf.get_variable_scope().reuse()
                (cell_output, state) = cell(inputs_embedding[:, time_step, :], state)
                outputs.append(cell_output)
            
            output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, hidden_size]) #[batch_size * num_steps, hidden_size]
            weight_o = tf.get_variable('output_weight', [hidden_size, vocab_size], 'float32')
            bias_o = tf.get_variable('output_bias', [vocab_size], 'float32')
            logits = tf.matmul(output, weight_o) + bias_o
            
            loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
                    [logits], [tf.reshape(ground_truth, [-1])],
                    [tf.ones([batch_size * num_steps], dtype='float32')])
            
            self.loss = tf.reduce_sum(loss) / batch_size

            if is_training:
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
            self.num_steps = num_steps
            # Output Module
            if is_training:
                self.opt_op = opt_op
            else:
                self.opt_op = None


    def get_feed_dict(self, batches, is_train):
        return {
            self.x: batches[0],
            self.y: batches[1],
            self.is_training: is_train
        }
    
    def save_params(self):
        assert self.action == 'train'
        params = self.params
        filename = os.path.join(self.save_dir, "params.json")
        save_params_dict = {'rnnlm_layers': params.rnnlm_layers,
                            'rnnlm_hidden_size': params.rnnlm_hidden_size,
                            'num_steps': self.num_steps,
                            'target': params.target,
                            'arch': params.arch,
                            'task': params.task}
        with open(filename, 'w') as file:
            json.dump(save_params_dict, file, indent=4)

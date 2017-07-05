import os
import json
import numpy as np
import tensorflow as tf

from functools import partial

from expert.base_model import BaseModel
from tf_helper.nn import prelu
from tf_helper.model_utils import get_sequence_length
from ren_helper.dynamic_memory_cell import DynamicMemoryCell

class REN(BaseModel):
    def build(self, forward_only):
        params = self.params
        batch_size, sentence_size, question_size, story_size = params.batch_size, params.sentence_size, params.question_size, params.story_size
        embedding_size, vocab_size = params.ren_embedding_size, self.words.vocab_size
        num_blocks = params.ren_num_blocks

        # initialize self
        # placeholders
        story = tf.placeholder('int32', shape=[batch_size, story_size, sentence_size], name='x')  # [num_batch, fact_count, sentence_len]
        question = tf.placeholder('int32', shape=[batch_size, question_size], name='q')  # [num_batch, question_len]
        answer = tf.placeholder('int32', shape=[batch_size], name='y')  # [num_batch] - one word answer
        is_training = tf.placeholder(tf.bool)
        # batch_size = tf.shape(story)[0]

        normal_initializer = tf.random_normal_initializer(stddev=0.1)
        ones_initializer = tf.constant_initializer(1.0)
        activation = partial(prelu, initializer=ones_initializer)

        with tf.variable_scope('EntityNetwork', initializer=normal_initializer):
            # Embeddings
            # The embedding mask forces the special "pad" embedding to zeros.
            embedding_params = tf.get_variable('embedding_params', [vocab_size, embedding_size])
            embedding_mask = tf.constant([0 if i == 0 else 1 for i in range(vocab_size)],
                dtype=tf.float32,
                shape=[vocab_size, 1])
            embedding_params_masked = embedding_params * embedding_mask


            story_embedding = tf.nn.embedding_lookup(embedding_params_masked, story)
            question_embedding = tf.nn.embedding_lookup(embedding_params_masked, tf.expand_dims(question, 1))

            # Input Module
            encoded_story = self.get_input_encoding(story_embedding, ones_initializer, 'StoryEncoding')
            encoded_question = self.get_input_encoding(question_embedding, ones_initializer, 'QuestionEncoding')

            # Memory Module
            # We define the keys outside of the cell so they may be used for state initialization.
            keys = [tf.get_variable('key_{}'.format(j), [embedding_size]) for j in range(num_blocks)]

            cell = DynamicMemoryCell(num_blocks, embedding_size, keys,
                initializer=normal_initializer,
                activation=activation)

            # Recurrence
            initial_state = cell.zero_state(batch_size, tf.float32)
            sequence_length = get_sequence_length(encoded_story)
            _, last_state = tf.nn.dynamic_rnn(cell, encoded_story,
                sequence_length=sequence_length,
                initial_state=initial_state)

            logits = self.get_output(last_state, encoded_question,
                                     num_blocks=num_blocks,
                                     vocab_size=vocab_size,
                                     initializer=normal_initializer,
                                     activation=activation)

            logits = tf.contrib.layers.batch_norm(logits, decay=0.9, is_training=is_training, center=True, scale=True,
                                                  updates_collections=None, scope='BatchNorm')

            self.ans_logits = logits
            self.output = tf.nn.softmax(logits)
            predicts = tf.argmax(logits, 1)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=answer)
            self.total_loss = tf.reduce_mean(cross_entropy)
            corrects = tf.equal(tf.cast(predicts, 'int32'), answer)
            num_corrects = tf.reduce_sum(tf.cast(corrects, tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))
            tf.summary.scalar('loss', self.total_loss, collections=['TRAIN_SUMM'])
            tf.summary.scalar('accuracy', self.accuracy, collections=['TRAIN_SUMM'])
            self.merged_TRAIN = tf.summary.merge_all(key='TRAIN_SUMM')

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
                opt_op = tf.contrib.layers.optimize_loss(self.total_loss,
                                                         self.global_step,
                                                         learning_rate=params.learning_rate,
                                                         optimizer=tf.train.AdamOptimizer,
                                                         clip_gradients=10.,
                                                         learning_rate_decay_fn=learning_rate_decay_fn,
                                                         summaries=OPTIMIZER_SUMMARIES)
            self.x = story
            self.q = question
            self.y = answer
            self.is_training = is_training

            # Output Module
            if not forward_only:
                self.opt_op = opt_op
            else:
                self.opt_op = None

    def get_input_encoding(self, embedding, initializer=None, scope=None):
        """
        Implementation of the learned multiplicative mask from Section 2.1, Equation 1. This module is also described
        in [End-To-End Memory Networks](https://arxiv.org/abs/1502.01852) as Position Encoding (PE). The mask allows
        the ordering of words in a sentence to affect the encoding.
        """
        with tf.variable_scope(scope, 'Encoding', initializer=initializer):
            _, _, max_sentence_length, _ = embedding.get_shape().as_list()
            positional_mask = tf.get_variable('positional_mask', [max_sentence_length, 1])
            encoded_input = tf.reduce_sum(embedding * positional_mask, axis=[2])
            return encoded_input

    def get_output(self, last_state, encoded_question, num_blocks, vocab_size,
            activation=tf.nn.relu,
            initializer=None,
            scope=None):
        """
        Implementation of Section 2.3, Equation 6. This module is also described in more detail here:
        [End-To-End Memory Networks](https://arxiv.org/abs/1502.01852).
        """
        with tf.variable_scope(scope, 'Output', initializer=initializer):
            last_state = tf.stack(tf.split(axis=1, num_or_size_splits=num_blocks, value=last_state), axis=1)
            _, _, embedding_size = last_state.get_shape().as_list()

            # Use the encoded_query to attend over memories (hidden states of dynamic last_state cell blocks)
            attention = tf.reduce_sum(last_state * encoded_question, axis=[2])

            # Subtract max for numerical stability (softmax is shift invariant)
            attention_max = tf.reduce_max(attention, axis=[-1], keep_dims=True)
            attention = tf.nn.softmax(attention - attention_max)
            attention = tf.expand_dims(attention, 2)

            # Weight memories by attention vectors
            u = tf.reduce_sum(last_state * attention, axis=[1])

            # R acts as the decoder matrix to convert from internal state to the output vocabulary size
            R = tf.get_variable('R', [embedding_size, vocab_size])
            H = tf.get_variable('H', [embedding_size, embedding_size])

            q = tf.squeeze(encoded_question, axis=[1])
            y = tf.matmul(activation(q + tf.matmul(u, H)), R)
            return y

    def get_feed_dict(self, batches, is_train):
        return {
            self.x: batches[0],
            self.q: batches[1],
            self.y: batches[2],
            self.is_training: is_train
        }
    
    def save_params(self):
        assert self.action == 'train'
        params = self.params
        filename = os.path.join(self.save_dir, "params.json")
        save_params_dict = {'ren_num_blocks': params.ren_num_blocks,
                            'ren_embedding_size': params.ren_embedding_size,
                            'target': params.target,
                            'arch': params.arch,
                            'task': params.task}
        with open(filename, 'w') as file:
            json.dump(save_params_dict, file, indent=4)

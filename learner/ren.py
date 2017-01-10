import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from functools import partial

from learner.base_model import BaseModel
from ren_helper.activations import prelu
from ren_helper.dynamic_memory_cell import DynamicMemoryCell
from ren_helper.model_utils import get_sequence_length


class REN(BaseModel):
    def build(self, forward_only):
        params = self.params
        batch_size, sentence_size, question_size, story_size = params.batch_size, params.sentence_size, params.question_size, params.story_size
        embedding_size, vocab_size = params.ren_embedding_size, self.words.vocab_size
        num_blocks = params.ren_num_blocks

        # initialize self
        # placeholders
        story = tf.placeholder('int32', shape=[None, story_size, sentence_size], name='x')  # [num_batch, fact_count, sentence_len]
        question = tf.placeholder('int32', shape=[None, question_size], name='q')  # [num_batch, question_len]
        answer = tf.placeholder('int32', shape=[None], name='y')  # [num_batch] - one word answer
        is_training = tf.placeholder(tf.bool)
        feed_previous = tf.placeholder(tf.bool) #feed_previous if testing
        # batch_size = tf.shape(input)[0]

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
            #question_embedding = tf.nn.embedding_lookup(embedding_params_masked, tf.expand_dims(question, 1))
            answer_embedding = tf.nn.embedding_lookup(embedding_params_masked, tf.expand_dims(answer, 1)) 
            # [num_batch, 1, embedding_size]
            
            # Input Module
            encoded_story = self.get_input_encoding(story_embedding, ones_initializer, 'StoryEncoding')
            #encoded_question = self.get_input_encoding(question_embedding, ones_initializer, 'QuestionEncoding')

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

            memory = self.get_memory(last_state, answer_embedding,
                num_blocks=num_blocks,
                initializer=normal_initializer,
                activation=activation) # [num_batch, embedding_size]

            ## decoder  ##
            
            ## output projection weight ##
            proj_w = tf.get_variable('proj_w', [embedding_size, vocab_size])
            proj_b = tf.get_variable('proj_b', vocab_size)
            ## build decoder inputs ##
            go_pad = tf.ones(shape = tf.pack([tf.shape(answer)[0], 1]), dtype = tf.int32)
            decoder_inputs = tf.concat(1, [go_pad, question])
            decoder_inputs = tf.nn.embedding_lookup(embedding_params_masked, decoder_inputs) 
            # [num_batch, Q+1, embedding_size]
            decoder_inputs = tf.transpose(decoder_inputs, [1, 0, 2]) # [Q+1, num_batch, embedding_size]
            decoder_inputs = tf.unstack(decoder_inputs)[:-1] # Q * [num_batch, embedding_size]
            decoder_inputs = [tf.concat(1, [de_inp, memory]) for de_inp in decoder_inputs]
            ## question module rnn cell ##
            q_cell = rnn_cell.GRUCell(embedding_size)
            ## decoder state init ##
            q_init_state = memory
            ## decoder loop function ##
            def _loop_fn(prev, i):
                prev = tf.matmul(prev, proj_w) + proj_b
                prev_symbol = tf.argmax(prev, 1)
                emb_prev = tf.nn.embedding_lookup(embedding_params_masked, prev_symbol)
                return tf.concat(1, [emb_prev, memory])
            ## decoder rnn##
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

           
            ## seq loss ##
            target_list = tf.unstack(tf.transpose(question))
            total_loss = tf.nn.seq2seq.sequence_loss(q_logprobs, target_list,
                                                     [tf.ones(shape = tf.pack([tf.shape(answer)[0],])
                                                              , dtype = tf.float32)] * question_size
                                                     )

            #self.output = tf.nn.softmax(logits)
            #predicts = tf.argmax(self.output, 1)
            #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, answer)
            #self.total_loss = tf.reduce_mean(cross_entropy)
            """
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
            """
            # placeholder
        self.x = story
        self.q = question
        self.y = answer
        self.is_training = is_training
        self.feed_previous = feed_previous
            #Tensors
        self.total_loss = total_loss
        self.output = q_probs

            # Output Module
        if forward_only:
            self.opt_op = None
        else:
            self.Pre_opt_op = self.PreTrainOpt()

    def get_input_encoding(self, embedding, initializer=None, scope=None):
        """
        Implementation of the learned multiplicative mask from Section 2.1, Equation 1. This module is also described
        in [End-To-End Memory Networks](https://arxiv.org/abs/1502.01852) as Position Encoding (PE). The mask allows
        the ordering of words in a sentence to affect the encoding.
        """
        with tf.variable_scope(scope, 'Encoding', initializer=initializer):
            _, _, max_sentence_length, _ = embedding.get_shape().as_list()
            positional_mask = tf.get_variable('positional_mask', [max_sentence_length, 1])
            encoded_input = tf.reduce_sum(embedding * positional_mask, reduction_indices=[2])
            return encoded_input

    def get_memory(self, last_state, answer_embedding, num_blocks,
            activation=tf.nn.relu,
            initializer=None,
            scope=None):
        """
        Implementation of Section 2.3, Equation 6. This module is also described in more detail here:
        [End-To-End Memory Networks](https://arxiv.org/abs/1502.01852).
        """
        with tf.variable_scope(scope, 'Memory', initializer=initializer):
            last_state = tf.pack(tf.split(1, num_blocks, last_state), axis=1)
            _, _, embedding_size = last_state.get_shape().as_list() # [num_batch, num_blocks, embedding_size]

            # Use the answer_embedding to attend over memories (hidden states of dynamic last_state cell blocks)
            attention = tf.reduce_sum(last_state * answer_embedding, reduction_indices=[2]) # [num_batch, num_blocks]

            # Subtract max for numerical stability (softmax is shift invariant)
            attention_max = tf.reduce_max(attention, reduction_indices=[-1], keep_dims=True) # [num_batch, 1]
            attention = tf.nn.softmax(attention - attention_max)
            attention = tf.expand_dims(attention, 2) # [num_batch, num_blocks, 1]

            # Weight memories by attention vectors
            u = tf.reduce_sum(last_state * attention, reduction_indices=[1]) # [num_batch, embedding_size] 

            # R acts as the decoder matrix to convert from internal state to the output vocabulary size
            #R = tf.get_variable('R', [embedding_size, vocab_size])
            H = tf.get_variable('H', [embedding_size, embedding_size])

            a = tf.squeeze(answer_embedding, squeeze_dims=[1]) # [num_batch, embedding_size]
            memory = activation(a + tf.matmul(u, H))
            #y = tf.matmul(activation(q + tf.matmul(u, H)), R)
            return memory

    def get_feed_dict(self, batches, feed_previous, is_train):
        return {
            self.x: batches[0],
            self.q: batches[1],
            self.y: batches[2],
            self.is_training: is_train,
            self.feed_previous : feed_previous
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

    def PreTrainOpt(self):
        with tf.name_scope("PreTrainOpt"):
            def learning_rate_decay_fn(lr, global_step):
                return tf.train.exponential_decay(lr,
                                                  global_step,
                                                  decay_steps=3000,
                                                  decay_rate=0.5,
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
            return opt_op



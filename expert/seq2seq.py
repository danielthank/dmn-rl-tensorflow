import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import static_bidirectional_rnn

from expert.base_model import BaseModel
from dmn_helper.nn import weight, bias, dropout, batch_norm, variable_summary, gumbel_softmax
from dmn_helper.episode_module import EpisodeModule
from ren_helper.model_utils import get_sequence_length


EPS = 1e-20
OPTIMIZER_SUMMARIES = ["learning_rate",
                       "loss",
                       "gradients",
                       "gradient_norm"]


class Seq2Seq(BaseModel):
    def build(self, forward_only):
        params = self.params
        L, Q, F = params.sentence_size, params.question_size, params.story_size
        V, d, A = params.dmn_embedding_size, params.dmn_embedding_size, self.words.vocab_size

        # initialize self
        # placeholders
        input = tf.placeholder('int32', shape=[None, F, L], name='x')
        question = tf.placeholder('int32', shape=[None, Q], name='q')
        answer = tf.placeholder('int32', shape=[None], name='y')
        self.batch_size = tf.shape(answer)[0]
        fact_counts = get_sequence_length(input)
        is_training = tf.placeholder(tf.bool)

        # Prepare parameters
        gru = rnn.GRUCell(d)
        l = self.positional_encoding()

        with tf.variable_scope('Embedding'):
            embedding = weight('embedding', [A, V], init='uniform', range=3**(1/2))
            embedding_mask = tf.constant([0 if i == 0 else 1 for i in range(A)], dtype=tf.float32, shape=[A, 1])
            embedding = embedding * embedding_mask
            variable_summary([embedding])

        with tf.name_scope('Tied_SentenceReader'):
            input_embed = tf.nn.embedding_lookup(embedding, input) # [batch, story, sentence] -> [batch, story, sentence, embedding_size]
            # apply positional encoding
            encoded = l * input_embed
            facts = tf.reduce_sum(encoded, 2)  # [batch, story, embedding_size]
            # dropout
            # facts = dropout(facts, params.dmn_keep_prob, is_training)

        with tf.variable_scope('Question_embed'):
            ques_embed = tf.nn.embedding_lookup(embedding, question) # [N, Q, V]
        
        with tf.variable_scope('InputFusion') as scope:
            (output_fw, output_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(gru,
                                                                             gru,
                                                                             facts,
                                                                             sequence_length=fact_counts,
                                                                             dtype=tf.float32)
            f_outputs = output_fw + output_bw
            q_init_state = state_fw + state_bw

        ## QA Branch
        with tf.variable_scope('QA_branch'):
            QA_ans_logits = self.QA_branch(gru, ques_embed, f_outputs, is_training)
            QA_ans = tf.nn.softmax(QA_ans_logits)

            with tf.name_scope('QA_Loss'):
                # Cross-Entropy loss
                QA_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=QA_ans_logits, labels=answer)
                QA_loss = tf.reduce_mean(QA_cross_entropy)
                QA_total_loss = QA_loss + params.dmn_weight_decay * tf.add_n(tf.get_collection('l2'))

            with tf.variable_scope('QA_Accuracy'):
                # Accuracy
                predicts = tf.cast(tf.argmax(QA_ans_logits, 1), 'int32')
                corrects = tf.equal(predicts, answer)
                num_corrects = tf.reduce_sum(tf.cast(corrects, tf.float32))
                accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

        # placeholders
        self.x = input
        self.q = question
        self.y = answer
        self.fc = fact_counts
        self.is_training = is_training

        # QA output tensors
        self.QA_ans = QA_ans
        self.total_loss = QA_total_loss
        self.num_corrects = num_corrects
        self.accuracy = accuracy

        # optimizer ops
        self.opt_op, QA_opt_sc = self.QAOpt()

    def QA_branch(self, gru, ques_embed, f_outputs, is_training):
        params = self.params
        sentence_size, question_size, story_size = params.sentence_size, params.question_size, params.story_size
        vocab_size = self.words.vocab_size
        embedding_size = params.dmn_embedding_size
        hidden_size = params.dmn_embedding_size

        decoder_input = tf.unstack(ques_embed, axis=1)
        init_state = tf.zeros(shape=[self.batch_size, hidden_size], dtype=tf.float32)
 
        _, state = tf.contrib.legacy_seq2seq.attention_decoder(decoder_input,
                                                               init_state,
                                                               f_outputs,
                                                               gru)
        W = tf.get_variable('hidden2vocab', shape=[hidden_size, vocab_size])
        ans_logits = tf.matmul(state, W)
        return ans_logits

    def QAOpt(self):
        with tf.variable_scope("QAOpt") as scope:
            def learning_rate_decay_fn(lr, global_step):
                return tf.train.exponential_decay(lr,
                                                  global_step,
                                                  decay_steps=5000,
                                                  decay_rate=0.95,
                                                  staircase=True)
            OPTIMIZER_SUMMARIES = ["learning_rate",
                                   "loss"]
            opt_op = tf.contrib.layers.optimize_loss(self.total_loss,
                                                     self.global_step,
                                                     learning_rate=self.params.learning_rate,
                                                     optimizer=tf.train.AdamOptimizer,
                                                     clip_gradients=5.,
                                                     learning_rate_decay_fn=learning_rate_decay_fn,
                                                     summaries=OPTIMIZER_SUMMARIES)
            for var in tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope.name):
                tf.add_to_collection("QA_SUMM", var)
            return opt_op, scope.name

    def positional_encoding(self):
        V, L = self.params.dmn_embedding_size, self.params.sentence_size
        encoding = np.zeros([L, V])
        for l in range(L):
            for v in range(V):
                encoding[l, v] = (1 - float(l)/L) - (float(v)/V)*(1 - 2.0*l/L)

        return encoding

    def get_feed_dict(self, batches,  is_train):
        return {
            self.x: batches[0],
            self.q: batches[1],
            self.y: batches[2],
            self.is_training: is_train,
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
                            'arch': params.arch,
                            'task': params.task}
        with open(filename, 'w') as file:
            json.dump(save_params_dict, file, indent=4)

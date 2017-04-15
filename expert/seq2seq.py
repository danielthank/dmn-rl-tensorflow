import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import static_bidirectional_rnn

from expert.base_model import BaseModel
from tf_helper.nn import weight, bias, dropout, batch_norm, variable_summary, gumbel_softmax
from tf_helper.model_utils import get_sequence_length


EPS = 1e-20
OPTIMIZER_SUMMARIES = ["learning_rate",
                       "loss",
                       "gradients",
                       "gradient_norm"]


class Seq2Seq(BaseModel):
    def build(self, forward_only):
        params = self.params
        L, Q, F = params.sentence_size, params.question_size, params.story_size
        V, A = params.seq2seq_hidden_size, self.words.vocab_size

        # initialize self
        # placeholders
        input = tf.placeholder('int32', shape=[None, F, L], name='x')
        question = tf.placeholder('int32', shape=[None, Q], name='q')
        answer = tf.placeholder('int32', shape=[None], name='y')
        self.batch_size = tf.shape(answer)[0]
        fact_counts = get_sequence_length(input)
        is_training = tf.placeholder(tf.bool)
        feed_previous = tf.placeholder(tf.bool)

        # Prepare parameters
        gru = rnn.GRUCell(V)
        story_positional_encoding = self.positional_encoding(L, V)
        question_positional_encoding = self.positional_encoding(Q, V)
        embedding_mask = tf.constant([0 if i == 0 else 1 for i in range(A)], dtype=tf.float32, shape=[A, 1])
        qa_embedding = tf.get_variable('qa_embedding', [A, V], initializer=tf.contrib.layers.xavier_initializer())
        qa_embedding = qa_embedding * embedding_mask

        qg_embedding = tf.get_variable('qg_embedding', [A, V], initializer=tf.contrib.layers.xavier_initializer())
        qg_embedding = qg_embedding * embedding_mask

        with tf.variable_scope('QA_SentenceReader'):
            qa_story = tf.nn.embedding_lookup(qa_embedding, input) # [batch, story, sentence] -> [batch, story, sentence, embedding_size]
            # apply positional encoding
            qa_story = story_positional_encoding * qa_story
            qa_story = tf.reduce_sum(qa_story, 2)  # [batch, story, embedding_size]
            qa_story = dropout(qa_story, 0.5, is_training)
            (qa_states_fw, qa_states_bw), (_, _) = tf.nn.bidirectional_dynamic_rnn(gru,
                                                                                   gru,
                                                                                   qa_story,
                                                                                   sequence_length=fact_counts,
                                                                                   dtype=tf.float32)
            qa_story = qa_states_fw + qa_states_bw

        with tf.name_scope('QA_QuestionReader'):
            qa_q = tf.nn.embedding_lookup(qa_embedding, question) # [N, Q, V]
            qa_q = question_positional_encoding * qa_q
            qa_q = tf.reduce_sum(qa_q, 1) # [N, V]
            qa_q = dropout(qa_q, 0.5, is_training)

        ## QA Branch
        with tf.variable_scope('QA_branch'):
            QA_ans_logits = self.QA_branch(qa_embedding, qa_q, qa_story, is_training)
            QA_ans = tf.nn.softmax(QA_ans_logits)

            with tf.name_scope('QA_Loss'):
                # Cross-Entropy loss
                QA_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=QA_ans_logits, labels=answer)
                QA_loss = tf.reduce_mean(QA_cross_entropy)
                # QA_total_loss = QA_loss + params.dmn_weight_decay * tf.add_n(tf.get_collection('l2'))
                QA_total_loss = QA_loss

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

    def QA_branch(self,embedding, qa_q, qa_story, is_training):
        params = self.params
        # attention mechanism
        q_cell = rnn.GRUCell(params.seq2seq_hidden_size)
        go_pad = tf.ones(tf.stack([self.batch_size, 1]), dtype=tf.int32)
        go_pad = tf.nn.embedding_lookup(embedding, go_pad) # [N, 1, V]
        decoder_inputs = tf.unstack(go_pad, axis=1) # 1 * [N, V]

        q_logprobs, _ = tf.contrib.legacy_seq2seq.attention_decoder(decoder_inputs=decoder_inputs,
                                                                    initial_state=qa_q,
                                                                    attention_states=qa_story,
                                                                    cell=q_cell,
                                                                    output_size=self.words.vocab_size,
                                                                    loop_function=None)
        return q_logprobs[0]

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

    def positional_encoding(self, sentence_size, embedding_size):
        encoding = np.zeros([sentence_size, embedding_size])
        for l in range(sentence_size):
            for v in range(embedding_size):
                encoding[l, v] = (1 - float(l)/sentence_size) - (float(v)/embedding_size)*(1 - 2.0*l/sentence_size)
        return encoding

    def def_run_list(self):
        self.QA_train_list  = [self.merged_QA, self.QA_opt_op, self.global_step, self.QA_total_loss, self.accuracy]
        self.QA_test_list   = [self.QA_total_loss, self.accuracy, self.global_step]

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
        save_params_dict = {'seq2seq_hidden_size': params.seq2seq_hidden_size,
                            'seq2seq_weight_decay': params.seq2seq_weight_decay,
                            'target': params.target,
                            'arch': params.arch,
                            'task': params.task}
        with open(filename, 'w') as file:
            json.dump(save_params_dict, file, indent=4)

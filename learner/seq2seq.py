import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import static_bidirectional_rnn

from learner.base_model import BaseModel
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
        feed_previous = tf.placeholder(tf.bool)

        # Prepare parameters
        gru = rnn.GRUCell(d)
        story_positional_encoding = self.positional_encoding(L, V)
        question_positional_encoding = self.positional_encoding(Q, V)

        embedding_mask = tf.constant([0 if i == 0 else 1 for i in range(A)], dtype=tf.float32, shape=[A, 1])
        qa_embedding = tf.get_variable('qa_embedding', [A, V], initializer=tf.contrib.layers.xavier_initializer())
        qa_embedding = qa_embedding * embedding_mask
        # variable_summary([qa_embedding])

        qg_embedding = tf.get_variable('qg_embedding', [A, V], initializer=tf.contrib.layers.xavier_initializer())
        qg_embedding = qg_embedding * embedding_mask

        with tf.variable_scope('QA_SentenceReader'):
            qa_story = tf.nn.embedding_lookup(qa_embedding, input) # [batch, story, sentence] -> [batch, story, sentence, embedding_size]
            # apply positional encoding
            qa_story = story_positional_encoding * qa_story
            qa_story = tf.reduce_sum(qa_story, 2)  # [batch, story, embedding_size]
            # dropout
            # facts = dropout(facts, params.dmn_keep_prob, is_training)
            (qa_states_fw, qa_states_bw), (_, _) = tf.nn.bidirectional_dynamic_rnn(gru,
                                                                                   gru,
                                                                                   qa_story,
                                                                                   sequence_length=fact_counts,
                                                                                   dtype=tf.float32)
            qa_story = qa_states_fw + qa_states_bw

        with tf.variable_scope('QG_SentenceReader'):
            qg_story = tf.nn.embedding_lookup(qg_embedding, input) # [batch, story, sentence] -> [batch, story, sentence, embedding_size]
            # apply positional encoding
            qg_story = story_positional_encoding * qg_story
            qg_story = tf.reduce_sum(qg_story, 2)  # [batch, story, embedding_size]
            # dropout
            # facts = dropout(facts, params.dmn_keep_prob, is_training)
            (qg_states_fw, qg_states_bw), (_, _) = tf.nn.bidirectional_dynamic_rnn(gru,
                                                                                   gru,
                                                                                   qg_story,
                                                                                   sequence_length=fact_counts,
                                                                                   dtype=tf.float32)
            qg_story = qg_states_fw + qg_states_bw

        with tf.name_scope('QA_QuestionReader'):
            qa_q = tf.nn.embedding_lookup(qa_embedding, question) # [N, Q, V]
            qa_q = question_positional_encoding * qa_q
            qa_q = tf.reduce_sum(qa_q, 1) # [N, V]

        with tf.name_scope('QG_QuestionReader'):
            qg_q = tf.nn.embedding_lookup(qg_embedding, question) # [N, Q, V]
            #qg_q = question_positional_encoding * qg_q
            #qg_q = tf.reduce_sum(qg_q, 1) # [N, V]
        
        ## QA Branch
        with tf.name_scope('QA_branch'):
            QA_ans_logits = self.QA_branch(qa_embedding, qa_q, qa_story, is_training)
            QA_ans = tf.nn.softmax(QA_ans_logits)

            with tf.name_scope('QA_Loss'):
                # Cross-Entropy loss
                QA_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=QA_ans_logits, labels=answer)
                QA_loss = tf.reduce_mean(QA_cross_entropy)
                #QA_total_loss = QA_loss + params.dmn_weight_decay * tf.add_n(tf.get_collection('l2'))
                QA_total_loss = QA_loss

            with tf.variable_scope('QA_Accuracy'):
                # Accuracy
                predicts = tf.cast(tf.argmax(QA_ans_logits, 1), 'int32')
                corrects = tf.equal(predicts, answer)
                num_corrects = tf.reduce_sum(tf.cast(corrects, tf.float32))
                accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

        ## QG Branch
        with tf.variable_scope('QG_branch') as scope:
            q_logprobs = self.QG_branch(qg_embedding, qg_q, qg_story, feed_previous)
            q_probs = [tf.nn.softmax(out) for out in q_logprobs]
            variables = [v for v in tf.trainable_variables() if v.name.startswith(scope.name)]
            variable_summary(variables)

            with tf.name_scope('QG_Loss'):
                target_list = tf.unstack(tf.transpose(question)) # Q * [N]
                # Cross-Entropy loss
                QG_loss = tf.contrib.legacy_seq2seq.sequence_loss(q_logprobs, target_list,
                                                   [tf.ones(shape=tf.stack([tf.shape(answer)[0],]), dtype=tf.float32)] * Q )
                #QG_total_loss = QG_loss + params.dmn_weight_decay * tf.add_n(tf.get_collection('l2'))
                QG_total_loss = QG_loss

        # Policy Gradient
        chosen_one_hot = tf.placeholder(tf.float32, shape=[None, Q, A], name='act')
        rewards = tf.placeholder(tf.float32, shape=[None], name='rewards')

        with tf.name_scope("PolicyGradient"):
            stack_q_probs = tf.stack(q_probs, axis=1) # Q * [N, A] -> [N, Q, A]
            act_probs = stack_q_probs * chosen_one_hot # [N, Q, A]
            act_probs = tf.reduce_prod(tf.reduce_sum(act_probs, axis=2), axis=1) # [N, Q, A] -> [N, Q] -> [N]

            # J = -1.*tf.reduce_mean(tf.log(act_probs+EPS)*rewards) + params.dmn_weight_decay*tf.add_n(tf.get_collection('l2'))
            J = -1.*tf.reduce_mean(tf.log(act_probs+EPS)*rewards) 


        # placeholders
        self.x = input
        self.q = question
        self.y = answer
        self.fc = fact_counts
        self.is_training = is_training
        self.feed_previous = feed_previous

        # policy gradient placeholders
        self.chosen_one_hot = chosen_one_hot
        self.rewards = rewards

        # QA output tensors
        self.QA_ans = QA_ans
        self.QA_total_loss = QA_total_loss
        self.num_corrects = num_corrects
        self.accuracy = accuracy

        # QG output tensors
        self.q_probs = q_probs
        self.QG_total_loss = QG_total_loss

        # policy gradient output tensors
        self.J = J

        # optimizer ops
        if not forward_only:
            self.RL_opt_op, RL_opt_sc = self.RLOpt()
            self.Pre_opt_op, Pre_opt_sc = self.PreOpt()
            self.QA_opt_op, QA_opt_sc = self.QAOpt()

    def QA_branch(self,embedding, qa_q, qa_story, is_training):
        params = self.params
        # attention mechanism
        q_cell = rnn.GRUCell(params.dmn_embedding_size)
        go_pad = tf.ones(tf.stack([self.batch_size, 1]), dtype=tf.int32)
        go_pad = tf.nn.embedding_lookup(embedding, go_pad) # [N, 1, V]
        decoder_inputs = tf.unstack(go_pad, axis=1) # 1 * [N, V]

        q_logprobs, _ = tf.contrib.legacy_seq2seq.attention_decoder(decoder_inputs=decoder_inputs,
                                                                    initial_state=qa_q,
                                                                    attention_states=qa_story,
                                                                    cell=q_cell,
                                                                    output_size=self.words.vocab_size,
                                                                    loop_function=None
                                                                    )
        return q_logprobs[0]
        """
        with tf.variable_scope('QA_attention'):
            W1 = tf.get_variable('hidden_w1', shape=[params.dmn_embedding_size, params.dmn_embedding_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            W2 = tf.get_variable('hidden_w2', shape=[params.dmn_embedding_size, params.dmn_embedding_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
            qa_story_w = tf.reshape(tf.matmul(tf.reshape(qa_story, [-1, params.dmn_embedding_size]), W1), [-1, params.story_size, params.dmn_embedding_size]) #[batch, story, hidden]
            qa_q_w = tf.matmul(qa_q, W2) #[batch, hidden]
            
            qa_all = qa_story_w + tf.reshape(qa_q_w, [-1, 1, params.dmn_embedding_size]) #[batch, story, hidden]
            qa_all = tf.nn.tanh(qa_all)
            v = tf.get_variable('v', shape=[params.dmn_embedding_size], initializer=tf.random_normal_initializer())
            score = tf.reduce_sum(v * qa_all, 2) #[batch, story]
            score = tf.nn.softmax(score)
            
            d = tf.reduce_sum(tf.reshape(score, [-1, params.story_size, 1]) * qa_story_w, 1) #[batch, hidden]

            W = tf.get_variable('hidden2vocab_w', shape=[params.dmn_embedding_size, self.words.vocab_size])
            b = tf.get_variable('hidden2vocab_b', shape=[self.words.vocab_size])

            ans_logits = tf.matmul(d, W) + b
            ans_logits = tf.nn.softmax(ans_logits)

        decoder_input = tf.unstack(ques_embed, axis=1)
        init_state = tf.zeros(shape=[self.batch_size, hidden_size], dtype=tf.float32)
 
        _, state = tf.contrib.legacy_seq2seq.embedding_attention_decoder(decoder_input,
                                                               init_state,
                                                               f_outputs,
                                                               gru)
        W = tf.get_variable('hidden2vocab', shape=[hidden_size, vocab_size])
        ans_logits = tf.matmul(state, W)
        return ans_logits
        """

    def QG_branch(self, embedding, qg_q, qg_story, feed_previous):
        params = self.params
        L, Q, F = params.sentence_size, params.question_size, params.story_size
        V, d, A = params.dmn_embedding_size, params.dmn_embedding_size, self.words.vocab_size
        ## output projection weight ##
        """
        proj_w = weight('proj_w', [parmas.dmn_embedding_size, self.words.vocab_size])
        proj_b = bias('proj_b', self.words.vocab_size)
        """
        ## build decoder inputs ##
        go_pad = tf.ones(tf.stack([self.batch_size, 1]), dtype=tf.int32)
        go_pad = tf.nn.embedding_lookup(embedding, go_pad) # [N, 1, V]
        decoder_inputs = tf.concat(axis=1, values=[go_pad, qg_q]) # [N, Q+1, V]
        decoder_inputs = tf.unstack(decoder_inputs, axis=1)[:-1] # Q * [N, V]

        ## question module rnn cell ##
        q_cell = rnn.GRUCell(params.dmn_embedding_size)

        ## decoder loop function ##
        def _loop_fn(prev, i):
            # prev = tf.matmul(prev, proj_w) + proj_b
            prev_symbol = tf.argmax(prev, 1)
            emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
            return emb_prev
        ## decoder ##
        def decoder(feed_previous_bool):
            loop_function = _loop_fn if feed_previous_bool else None
            reuse = None if feed_previous_bool else True
            with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                q_logprobs, _ = tf.contrib.legacy_seq2seq.attention_decoder(decoder_inputs=decoder_inputs,
                                                                            initial_state=qg_story[:, -1],
                                                                            attention_states=qg_story,
                                                                            cell=q_cell,
                                                                            output_size=self.words.vocab_size,
                                                                            loop_function=loop_function
                                                                            )
                return q_logprobs

        return tf.cond(feed_previous,
                        lambda: decoder(True),
                        lambda: decoder(False))
        # q_logprobs = [tf.matmul(out, proj_w) + proj_b for out in q_outputs]
        # return q_outputs


    def PreOpt(self):
        with tf.variable_scope("PreOpt") as scope:
            def learning_rate_decay_fn(lr, global_step):
                return tf.train.exponential_decay(lr,
                                                  global_step,
                                                  decay_steps=100,
                                                  decay_rate=0.95,
                                                  staircase=True)
            OPTIMIZER_SUMMARIES = ["learning_rate",
                                   "loss"]
            opt_op = tf.contrib.layers.optimize_loss(0.5*self.QA_total_loss + 0.5*self.QG_total_loss,
                                                     self.global_step,
                                                     learning_rate=self.params.learning_rate,
                                                     optimizer=tf.train.AdamOptimizer,
                                                     clip_gradients=1.,
                                                     learning_rate_decay_fn=learning_rate_decay_fn,
                                                     summaries=OPTIMIZER_SUMMARIES)
            for var in tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope.name):
                tf.add_to_collection("PRE_SUMM", var)
            return opt_op, scope.name

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
            opt_op = tf.contrib.layers.optimize_loss(self.QA_total_loss,
                                                     self.global_step,
                                                     learning_rate=self.params.learning_rate,
                                                     optimizer=tf.train.AdamOptimizer,
                                                     clip_gradients=5.,
                                                     learning_rate_decay_fn=learning_rate_decay_fn,
                                                     summaries=OPTIMIZER_SUMMARIES)
            for var in tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope.name):
                tf.add_to_collection("QA_SUMM", var)
            return opt_op, scope.name

    def RLOpt(self):
        with tf.variable_scope("RLOpt") as scope:
            def learning_rate_decay_fn(lr, global_step):
                return tf.train.exponential_decay(lr,
                                                  global_step,
                                                  decay_steps=5000,
                                                  decay_rate=0.95,
                                                  staircase=True)
            OPTIMIZER_SUMMARIES = ["learning_rate",
                                   "loss"]
            opt_op = tf.contrib.layers.optimize_loss(self.J,
                                                     self.global_step,
                                                     learning_rate=self.params.learning_rate,
                                                     optimizer=tf.train.AdamOptimizer,
                                                     clip_gradients=5.,
                                                     learning_rate_decay_fn=learning_rate_decay_fn,
                                                     summaries=OPTIMIZER_SUMMARIES)
            for var in tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope.name):
                tf.add_to_collection("RL_SUMM", var)
            return opt_op, scope.name

    def positional_encoding(self, sentence_size, embedding_size):
        encoding = np.zeros([sentence_size, embedding_size])
        for l in range(sentence_size):
            for v in range(embedding_size):
                encoding[l, v] = (1 - float(l)/sentence_size) - (float(v)/embedding_size)*(1 - 2.0*l/sentence_size)

        return encoding

    def get_feed_dict(self, batches, feed_previous, is_train):
        return {
            self.x: batches[0],
            self.q: batches[1],
            self.y: batches[2],
            self.is_training: is_train,
            self.feed_previous: feed_previous
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
                            'target': params.target,
                            'arch': params.arch,
                            'task': params.task}
        with open(filename, 'w') as file:
            json.dump(save_params_dict, file, indent=4)

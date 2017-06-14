import os
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import static_bidirectional_rnn
from tensorflow.contrib.layers import fully_connected, variance_scaling_initializer

from learner.base_model import BaseModel
from tf_helper.nn import weight, bias, dropout, batch_norm, variable_summary
from tf_helper.nn import gumbel_softmax, attention_decoder, create_opt
from tf_helper.model_utils import get_sequence_length, positional_encoding

from functools import reduce
from data_helper.question_memory import QuestionMemory
from data_helper.DQN_memory import DQNMemory
EPS = 1e-20


class TypeSelect(BaseModel):
    def __init__(self,words,params,*args):
        ## words ##
        self.words = words

        ## dirs ##
        self.save_dir = params.save_dir
        self.load_dir = params.load_dir
        if params.action == 'train':
            self.summary_dir = os.path.join(self.save_dir, 'pretrain_summary')
            self.validation_summary_dir = os.path.join(self.save_dir, 'pretrain_validation_summary')
            self.var_summary_dir = os.path.join(self.save_dir, 'pretrain_var_summary')
        elif params.action == 'rl':
            self.summary_dir = os.path.join(self.save_dir, 'RL_summary')
            self.validation_summary_dir = os.path.join(self.save_dir, 'RL_validation_summary')
            self.var_summary_dir = os.path.join(self.save_dir, 'RL_var_summary')

        ## set params ##
        self.action = params.action
        self.params = params

        ## build model graph ##
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.params.gpu_fraction)
        self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options))
        with self.graph.as_default():
            print("Building Learner model...")
            ## global step ##
            self.D_global_step = tf.Variable(0, name='D_global_step', trainable=False)
            self.Pre_global_step = tf.Variable(0, name='Pre_global_step', trainable=False)
            self.QA_global_step = tf.Variable(0, name='QA_global_step', trainable=False)
            self.RL_global_step = tf.Variable(0, name='RL_global_step', trainable=False)
            ## validation loss ##
            self.min_validation_loss = tf.Variable(np.inf, name='validation_loss', trainable=False)
            self.new_validation_loss = tf.placeholder('float32', name='new_validation_loss');
            self.assign_min_validation_loss = self.min_validation_loss.assign(self.new_validation_loss).op
            if not self.action == 'test':
                self.build(forward_only=False)
            else:
                self.build(forward_only=True)
            self.init_op = tf.global_variables_initializer()

        ## init saver ##
        with self.graph.as_default():
            self.saver = tf.train.Saver()

        ## init variables ##
        if not self.load_dir == '':
            print("Loading model ...")
            self.load()
        else:
            print("Init model ...")
            self.sess.run(self.init_op)

        ## summary writer##
        if not self.action == 'test':
            self.summary_writer = tf.summary.FileWriter(logdir=self.summary_dir, graph=self.sess.graph)
            self.validation_summary_writer = tf.summary.FileWriter(logdir=self.validation_summary_dir,
                                                                   graph=self.sess.graph)
            self.var_summary_writer = tf.summary.FileWriter(logdir=self.var_summary_dir, graph=self.sess.graph)

        ## reward baseline ##
        self.baseline = 0.

        ## define session run lists ##
        self.def_run_list()

    def build(self, forward_only):
        self.DQN_global_step = tf.Variable(0,name='DQN_global_Step',trainable=False)
        params = self.params
        L, Q, F = params.sentence_size, params.question_size, params.story_size
        V, A = params.seq2seq_hidden_size, self.words.vocab_size

        input = tf.placeholder('int32', shape=[None, F, L], name='x')
        question = tf.placeholder('int32', shape=[None, Q], name='q')
        answer = tf.placeholder('int32', shape=[None], name='y')
        self.batch_size = tf.shape(answer)[0]
        fact_counts = get_sequence_length(input)
        self.is_training = tf.placeholder(tf.bool)
        self.is_sample = tf.placeholder(tf.bool)
        feed_previous = tf.placeholder(tf.bool)
        story_positional_encoding = positional_encoding(L, V)
        question_positional_encoding = positional_encoding(Q, V)
        embedding_mask = tf.constant([0 if i == 0 else 1 for i in range(A)], dtype=tf.float32, shape=[A, 1])

        with tf.variable_scope('QA', initializer=tf.contrib.layers.xavier_initializer()):
            qa_embedding = tf.get_variable('qa_embedding', [A, V], initializer=tf.contrib.layers.xavier_initializer())
            qa_embedding = qa_embedding * embedding_mask
            with tf.variable_scope('SentenceReader'):
                qa_story = tf.nn.embedding_lookup(qa_embedding, input) # [batch, story, sentence] -> [batch, story, sentence, embedding_size]
                # apply positional encoding
                qa_story = story_positional_encoding * qa_story
                qa_story = tf.reduce_sum(qa_story, 2)  # [batch, story, embedding_size]
                qa_story = dropout(qa_story, 0.5, self.is_training)
                (qa_states_fw, qa_states_bw), (_, _) = tf.nn.bidirectional_dynamic_rnn(rnn.LSTMCell(V),
                                                                                       rnn.LSTMCell(V),
                                                                                       qa_story,
                                                                                       sequence_length=fact_counts,
                                                                                       dtype=tf.float32)
                qa_story = qa_states_fw + qa_states_bw
            with tf.name_scope('QuestionReader'):
                qa_q = tf.nn.embedding_lookup(qa_embedding, question) # [N, Q, V]
                qa_q = question_positional_encoding * qa_q
                qa_q = tf.reduce_sum(qa_q, 1) # [N, V]
                qa_q = dropout(qa_q, 0.5, self.is_training)

            QA_ans_logits = self.QA_branch(qa_embedding, qa_q, qa_story)
            QA_ans = tf.nn.softmax(QA_ans_logits)
            #variables = [v for v in tf.trainable_variables() if v.name.startswith(scope.name)]
            #variable_summary(variables)
            with tf.name_scope('Loss'):
                # Cross-Entropy loss
                QA_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=QA_ans_logits, labels=answer)
                QA_loss = tf.reduce_mean(QA_cross_entropy)
                #QA_total_loss = QA_loss + params.seq2seq_weight_decay * tf.add_n(tf.get_collection('l2'))
                QA_total_loss = QA_loss
            with tf.variable_scope('Accuracy'):
                # Accuracy
                predicts = tf.cast(tf.argmax(QA_ans_logits, 1), 'int32')
                corrects = tf.equal(predicts, answer)
                num_corrects = tf.reduce_sum(tf.cast(corrects, tf.float32))
                QA_accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))
            tf.summary.scalar('loss', QA_total_loss, collections=["QA_SUMM"])
            tf.summary.scalar('accuracy', QA_accuracy, collections=["QA_SUMM"])

        with tf.variable_scope('type_selector', initializer=tf.contrib.layers.xavier_initializer()):
            all_QA_vars = [x for x in tf.trainable_variables() if x.name.startswith('QA')]
            QA_decoder_vars = [x for x in tf.trainable_variables() if x.name.startswith('QA/attention_decoder')]
            for var in QA_decoder_vars:
                print ('name: ',var.name)
                print ('shape: ',var.shape)
            #sys.exit()
            QA_decoder_var_num = sum([reduce((lambda x,y:x*y),var.shape.as_list()) for var in QA_decoder_vars])
            print ('QA_var_num:%d'%QA_var_num)
            with tf.name_scope('DQN') :
                state = tf.placeholder(tf.float32,shape=[None,QA_deocder_var_num],name='leaner_state')
                actions = tf.placeholder(tf.float32,shape=[None,21],name='learner_actions')
                target_Q_value = tf.placeholder(tf.float32,shape=[None],name='target_Q_values')
                
                Q_values = tf.layers.dense(state,21) 
                action_Q_value = tf.reduce_sum(tf.multiply(Q_values,actions),reduction_indices=1)
                max_Q_value = tf.reduce_max(Q_values,axis=-1)
            
            with tf.name_scope('Loss'):
                DQN_loss = tf.reduce_mean(tf.square(action_Q_value-target_Q_value))
            
            tf.summary.scalar('DQN_loss',DQN_loss,collections=['type_selector']) 
        
        # placeholders
        self.x = input
        self.q = question
        self.y = answer
        self.fc = fact_counts
        self.feed_previous = feed_previous

        # type selector placeholders
        self.state = state
        self.actions = actions
        self.target_Q_value = target_Q_value

        # type selector output tensors
        self.Q_values = Q_values
        self.max_Q_value = max_Q_value
        self.DQN_loss = DQN_loss 

        # variables of QA
        self.all_QA_vars = all_QA_vars
        self.QA_decoder_vars = QA_decoder_vars
        self.QA_decoder_var_num = QA_decoder_var_num
        
        # QA output tensors
        self.QA_ans_logits = QA_ans_logits
        self.QA_ans = QA_ans
        self.QA_total_loss = QA_total_loss
        self.num_corrects = num_corrects
        self.QA_accuracy = QA_accuracy

        # optimizer ops
        if not forward_only:
            rl_l_rate = self.params.rl_learning_rate
            l_rate = self.params.learning_rate
            self.QA_opt_op = create_opt('QA_opt', self.QA_total_loss, l_rate, self.QA_global_step)
            self.DQN_opt_op = create_opt('DQN_opt',self.DQN_loss,l_rate,self.DQN_global_step)

        # merged summary ops
        self.merged_QA = tf.summary.merge_all(key='QA_SUMM')
        self.merged_VAR = tf.summary.merge_all(key='VAR_SUMM')
        self.merged_DQN = tf.summary.merge_all(key='DQN_SUMM')
    
    def train(self, train_data,val_data):    
        params = self.params
        assert self.action is not 'test'
        num_epoch = params.num_epochs
        max_action = 2000
        episode = 100
        epsilon = 1
        memory_size = 10000 
        sample_num = 1
        ## QA training memory, should be reset every episoode
        QA_x_mem = QuestionMemory((params.story_size, params.sentence_size), max_action*sample_num, dtype='int32')
        QA_q_mem = QuestionMemory((params.question_size,), max_action*sample_num, dtype='int32')
        QA_y_mem = QuestionMemory((), max_action*sample_num, dtype='int32')
        ## DQN memory pool
        dqn_memory_pool = DQNMemory((self.QA_decoder_var_num,),21,memory_size)
        for episo in range(episode):
            ## reset QA memory
            QA_x_mem.reset()
            QA_q_mem.reset()
            QA_y_mem.reset()
            
            last_state = None
            last_action = None
            for it in range(max_action+1):
                ## get learner QA variables value(learner state)
                QA_variables_value = self.sess.run(self.QA_vars)
                learner_state = np.zeros(0,'float32')
                for value in QA_variables_value:
                    learner_state = np.append(learner_state,value)
                
                ## get action Q value 
                Q_values = self.sess.run(self.Q_values,feed_dict = {self.state:np.expand_dims(learner_state,axis=0)})
                
                ## decide action
                if random.random() < epsilon: 
                    action = random.randint(0,20)
                else:
                    action = np.argmax(Q_values)
                
                if action == 20 or it == max_action:
                    ## train QA until converge
                    for _ in range(10): 
                        QA_summ,QA_global_step = self.QA_train(QA_x_mem,QA_q_mem,QA_y_mem,num_batch = 'all')
                        #self.summary_writer.add_summary(QA_summ, QA_global_step)
                    ## get rewords

                    reword = 
                    terminate = 0
                    
                else: 
                    ## choose training data from data set and add chosen data to QA memory
                    contexts,questions,ans = train_data[action].get_random_cnt(sample_num)
                    QA_X_mem.append(contexts)
                    QA_q_mem.append(questions)
                    QA_y_mem.append(ans)
                
                    ## train QA
                    QA_summ,QA_global_step = self.QA_train(QA_x_mem,QA_q_mem,QA_y_mem,num_batch = 'all')
                    #self.summary_writer.add_summary(QA_summ, QA_global_step)

                
                    ## get rewords
                    reword = 0 
                    terminate = 1
               
                ## push observation into DQN memory pool
                if last_state != None:
                    assert last_action != None
                    dqn_memory_pool.append(last_state,last_action,reword,state,terminate)
                last_state = state
                last_action = action
                
                ## memory replay
                self.DQN_train(dqn_memory_pool,num_batch = 1)
            ## reset QA model
            self.sess.run(tf.variable_initializer(self.all_QA_vars))
    def DQN_train(dqn_memory_pool,num_batch):
        params = self.params
        assert not self.action == 'test'
        
        index = np.arange(dqn_memory_pool)
        for i in range(num_batch):
            chosen_memory = np.random.choice(index,params.batch_size) 
            states,one_hot_actions,rewords,next_states,terminates = dqn_memory_pool[chosen_memory]
            
            ## get max Q value of next state
            max_Q_value = self.sess.run(self.max_Q_value,feed_dict = {self.state:next_states})
            target_Q_value = rewords + max_Q_value * terminates
            DQN_summ,QDN_global_step,_ = self.sess.run(self.DQN_train_list,
                                                       feed_dict = {self.state:state,
                                                                    self.actions:actions,
                                                                    self.target_Q_value:target_Q_value})
            self.summary_writer.add_summary(DQN_summ, DQN_global_step)
    
    def Discriminator(self, D_embedding, question):
        params = self.params 
        Q = params.question_size
        V = params.seq2seq_hidden_size
        filter_sizes = [1, 3]
        num_filters = [3, 5]
        
        with tf.name_scope('D_QuestionReader'):
            D_q = tf.nn.embedding_lookup(D_embedding, question) # [N, Q, V]
            #D_q = tf.expand_dims(D_q, -1) # [N, Q, V, 1]
            D_qf = tf.reshape(D_q, shape=[-1, Q*V]) # [N, Q, V] -> [N, Q*V]
        #[filter_height, filter_width, in_channels, out_channels]
        """
        pooled = []
        for filter_size, num_filter in zip(filter_sizes, num_filters):
            with tf.variable_scope("conv-max_%s" % filter_size):
                
                weight_initializer = variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=False)
                bias_initializer = variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
                conv = tf.contrib.layers.convolution2d(inputs = D_q,
                                                num_outputs = num_filter,
                                                kernel_size = [filter_size, V],
                                                stride = [1, 1],
                                                padding = 'VALID',
                                                activation_fn = tf.nn.relu,
                                                weights_initializer = weight_initializer,
                                                biases_initializer = bias_initializer) # [N, Q-k+1, 1, num_filter]
                pool = tf.contrib.layers.max_pool2d(inputs = conv,
                                                    kernel_size = [Q - filter_size + 1, 1],
                                                    stride = [1, 1],
                                                    padding = 'valid') #[N, 1, 1, num_filter]
                pooled.append(pool)
        D_qf = tf.reshape(tf.concat(pooled, 3), [-1, sum(num_filters)]) #[N, num_filters]
        """
        with tf.variable_scope("fc_layers"):
            weight_initializer = variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=False)
            bias_initializer = variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
            stack_args = [[32, tf.nn.relu],
                          [2, None]]
            D_logits = tf.contrib.layers.stack(D_qf,
                                               fully_connected,
                                               stack_args,
                                               weights_initializer=weight_initializer,
                                               biases_initializer=bias_initializer,
                                               scope="fc_layers")
        return D_logits
            


    def QA_branch(self, embedding, qa_q, qa_story):
        params = self.params
        # attention mechanism
        #q_cell = rnn.GRUCell(params.seq2seq_hidden_size)
        q_cell = rnn.LSTMCell(params.seq2seq_hidden_size)
        #q_cell = tf.contrib.rnn.MultiRNNCell([q_cell for l in range(num_layers)])
        go_pad = tf.ones(tf.stack([self.batch_size, 1]), dtype=tf.int32)
        go_pad = tf.nn.embedding_lookup(embedding, go_pad) # [N, 1, V]
        decoder_inputs = tf.unstack(go_pad, axis=1) # 1 * [N, V]

        q_logprobs, _ = tf.contrib.legacy_seq2seq.attention_decoder(decoder_inputs=decoder_inputs,
                                                                    initial_state=(qa_q, qa_q),
                                                                    attention_states=qa_story,
                                                                    cell=q_cell,
                                                                    output_size=self.words.vocab_size,
                                                                    loop_function=None)
        q_logprobs[0] = tf.contrib.layers.batch_norm(q_logprobs[0], decay=0.9, is_training=self.is_training, center=True, scale=True,
                                                     updates_collections=None, scope='BatchNorm')

        return q_logprobs[0]

    def QG_branch(self, embedding, qg_q, qg_story, feed_previous, is_training):
        params = self.params
        L, Q, F = params.sentence_size, params.question_size, params.story_size
        V, A = params.seq2seq_hidden_size, self.words.vocab_size
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

        ## output idxs ##
        chosen_idxs = []

        ## question module rnn cell ##
        #q_cell = rnn.GRUCell(params.seq2seq_hidden_size)
        q_cell = rnn.LSTMCell(params.seq2seq_hidden_size)
        #q_cell = tf.contrib.rnn.MultiRNNCell([q_cell for l in range(num_layers)])

        INIT_EPS = tf.constant(0.5, dtype='float32')
        FIN_EPS = tf.constant(0.5, dtype='float32')
        EXPLORE = tf.constant(500e3, dtype='float32')
        f32_RL_step = tf.cast(self.RL_global_step, 'float32')
        f32_Pre_step = tf.cast(self.Pre_global_step, 'float32')
        explore_eps =  tf.case({f32_Pre_step > f32_RL_step: (lambda: INIT_EPS),
                                EXPLORE <= (f32_RL_step - f32_Pre_step): (lambda: FIN_EPS)},
                               default=(lambda: INIT_EPS - (INIT_EPS - FIN_EPS) * (f32_RL_step - f32_Pre_step) / EXPLORE),
                               exclusive=True)
        tf.summary.scalar("explore_eps", explore_eps, collections=["VAR_SUMM"])
        ## decoder loop function ##
        def _loop_fn(prev, i):
            # prev = tf.matmul(prev, proj_w) + proj_b
            prev_symbol = tf.cond(self.is_sample,#tf.logical_and(is_training, feed_previous),
                                  lambda: gumbel_softmax(prev / explore_eps, 1),
                                  lambda: tf.argmax(prev, 1))
            chosen_idxs.append(prev_symbol)
            emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
            next_inp = tf.cond(feed_previous,
                               lambda: emb_prev,
                               lambda: decoder_inputs[i])
            return next_inp
        ## decoder ##
        def decoder(feed_previous_bool):
            loop_function = _loop_fn if feed_previous_bool else None
            reuse = None if feed_previous_bool else True
            with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                q_logprobs, _ = tf.contrib.legacy_seq2seq.attention_decoder(decoder_inputs=decoder_inputs,
                                                                            initial_state=(qg_story[:, -1], qg_story[:, -1]),
                                                                            attention_states=qg_story,
                                                                            cell=q_cell,
                                                                            output_size=self.words.vocab_size,
                                                                            loop_function=loop_function)
                q_logprobs = tf.contrib.layers.batch_norm(q_logprobs, decay=0.9, is_training=self.is_training, center=True, scale=True,
                                                          updates_collections=None, scope='BatchNorm')
                return q_logprobs
        """
        q_logprobs = tf.cond(feed_previous,
                             lambda: decoder(True),
                             lambda: decoder(False))
        """
        q_logprobs = decoder(True)

        last_symbol = tf.cond(self.is_sample,#tf.logical_and(is_training, feed_previous),
                              lambda: gumbel_softmax(q_logprobs[-1] / explore_eps, 1),
                              lambda: tf.argmax(q_logprobs[-1], 1))
        chosen_idxs.append(last_symbol)
        assert len(chosen_idxs) == Q
        return q_logprobs, chosen_idxs
        # q_logprobs = [tf.matmul(out, proj_w) + proj_b for out in q_outputs]
        # return q_outputs

    def def_run_list(self):
        #self.pre_train_list = [self.merged_QA, self.merged_QG, self.Pre_global_step, self.Pre_opt_op]
        self.QA_train_list  = [self.merged_QA, self.QA_global_step, self.QA_opt_op]
        #self.rl_train_list  = [self.merged_RL, self.RL_global_step, self.RL_opt_op]
        #self.D_train_list   = [self.merged_D, self.D_global_step, self.D_opt_op]

        #self.pre_test_list  = [self.merged_QA, self.merged_QG, self.Pre_global_step,
        #                       0.5*self.QA_total_loss+0.5*self.QG_total_loss]
        #self.rl_test_list   = [self.merged_QA, self.merged_QG, self.QA_global_step, self.QA_total_loss]

        self.DQN_train_list = [self.merged_DQN,self.DQN_global_step,self.DQN_opt_op]
    def get_feed_dict(self, batches, feed_previous, is_train, is_sample):
        return {
            self.x: batches[0],
            self.q: batches[1],
            self.y: batches[2],
            self.is_training: is_train,
            self.is_sample: is_sample,
            self.feed_previous: feed_previous
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

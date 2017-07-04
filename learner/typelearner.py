import os
import json
import random
import numpy as np
import math
import pickle
import gc
#gc.set_debug(gc.DEBUG_STATS)

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
        '''
        if params.action == 'train':
            self.summary_dir = os.path.join(self.save_dir, 'train_summary')
            self.validation_summary_dir = os.path.join(self.save_dir, 'train_validation_summary')
            self.var_summary_dir = os.path.join(self.save_dir, 'train_var_summary')
        self.test_summary_dir = os.path.join(self.save_dir, 'test_summary')
        '''
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
            self.reQA_global_step = tf.Variable(0, name='reQA_global_step', trainable=False)
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
        #if not self.action == 'test':
        self.task_summary_writers = {'train':[],'test':[]}
        for i in range(params.action_num):
            if i != params.action_num :
                task_summary_dir = os.path.join(self.save_dir,'task_%d'%(i+1))
                test_task_summary_dir = os.path.join(self.save_dir,'test_task_%d'%(i+1))
            else:
                task_summary_dir = os.path.join(self.save_dir,'terminate')
                test_task_summary_dir = os.path.join(self.save_dir,'test_terminate')
            self.task_summary_writers['train'].append(tf.summary.FileWriter(logdir=task_summary_dir,graph=self.sess.graph))
            self.task_summary_writers['test'].append(tf.summary.FileWriter(logdir=test_task_summary_dir,graph=self.sess.graph))
        
        self.summary_dir = os.path.join(self.save_dir, 'train_summary')
        self.test_summary_dir = os.path.join(self.save_dir, 'test_summary')
        self.validation_summary_dir = os.path.join(self.save_dir, 'train_validation_summary')
        self.var_summary_dir = os.path.join(self.save_dir, 'train_var_summary')
        
        self.summary_writer = tf.summary.FileWriter(logdir=self.summary_dir,graph=self.sess.graph)
        self.test_summary_writer = tf.summary.FileWriter(logdir=self.test_summary_dir,graph=self.sess.graph)
        self.validation_summary_writer = tf.summary.FileWriter(logdir=self.validation_summary_dir,
                                                               graph=self.sess.graph)
        self.var_summary_writer = tf.summary.FileWriter(logdir=self.var_summary_dir, graph=self.sess.graph)

        ## reward baseline ##
        self.baseline = 0.

        ## define session run lists ##
        self.def_run_list()
        self.initial = False
        
    def build(self, forward_only):
        self.DQN_global_step = tf.Variable(0,name='DQN_global_step',trainable=False)
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
                (qa_states_fw, qa_states_bw), (_, _) = tf.nn.bidirectional_dynamic_rnn(rnn.GRUCell(V),
                                                                                       rnn.GRUCell(V),
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
            QA_state_vars = [x for x in tf.trainable_variables() if x.name.startswith('QA')]
            variable_summary(QA_state_vars,["QA_VAR_SUMM"])

        with tf.variable_scope('type_selector', initializer=tf.contrib.layers.xavier_initializer()) :
            all_QA_vars = [x for x in tf.trainable_variables() if x.name.startswith('QA')]
            QA_state_vars = [x for x in tf.trainable_variables() if x.name.startswith('QA/qa_embedding')]
            for var in all_QA_vars:
                print ('name: ',var.name)
                print ('shape: ',var.shape)
            #sys.exit()
            QA_state_var_num = sum([reduce((lambda x,y:x*y),var.shape.as_list()) for var in QA_state_vars])
            print ('QA_var_num:%d'%QA_state_var_num)
            self.QA_var_reset = tf.variables_initializer(all_QA_vars)

            with tf.name_scope('DQN') as scope :
                ## DQN setting 
                self.state_mode = 'QA_var not_improve'
                self.other_state_size = 0
                self.var_state_size = 0
                if 'QA_var' in self.state_mode:
                    self.var_state_size += QA_state_var_num
                
                if 'action_record' in self.state_mode:
                    #self.other_state_size += params.action_num + 1
                    self.other_state_size += params.action_num

                if 'last_action' in self.state_mode:
                    #self.other_state_size += params.action_num + 1
                    self.other_state_size += params.action_num

                if 'last_reward' in self.state_mode:
                    self.other_state_size += 1

                if 'not_improve' in self.state_mode:
                    self.other_state_size += params.action_num
                
                if 'acc' in self.state_mode:
                    self.other_state_size += 1
                
                print ('state mode : %s'%self.state_mode)
                print ('var state size : %d'%self.var_state_size)
                print ('other state size : %d'%self.other_state_size)
               
                var_state = tf.placeholder(tf.float32,shape=[None,self.var_state_size],name='var_state')
                self.var_state_summary = tf.summary.histogram('var_state',var_state)
                
                other_state = tf.placeholder(tf.float32,shape=[None,self.other_state_size],name='other_state')
                actions = tf.placeholder(tf.float32,shape=[None,params.action_num],name='learner_actions')
                target_Q_value = tf.placeholder(tf.float32,shape=[None],name='target_Q_values')
               
                with tf.name_scope('var_state_function'):
                    small_var_state = tf.layers.dense(var_state,1024,activation=tf.nn.relu,name='state_function_1')
                    small_var_state = tf.layers.dense(small_var_state,1024,activation=tf.nn.relu,name='state_function_2')
                    small_var_state = tf.layers.dense(small_var_state,1024,name='state_function_output')
                
                with tf.name_scope('var_state_reconstruct'):
                    reconstruct = tf.layers.dense(small_var_state,1024,activation=tf.nn.relu,name='reconstruct_1')
                    reconstruct = tf.layers.dense(reconstruct,1024,activation=tf.nn.relu,name='reconstruct_2')
                    reconstruct = tf.layers.dense(reconstruct,self.var_state_size,name='reconstruct_output')
               
                small_var_state = tf.layers.dense(small_var_state,128,activation=tf.nn.relu)
                if self.var_state_size != 0 and self.other_state_size != 0:
                    merge = tf.concat([small_var_state,other_state],1)
                elif self.var_state_size != 0:
                    merge = small_var_state
                elif self.other_state_size != 0:
                    merge = other_state
                
                fc = tf.layers.dense(merge,128,activation=tf.nn.relu)
                Q_values = tf.layers.dense(fc,params.action_num,name='Q_value_output') 

                action_Q_value = tf.reduce_sum(tf.multiply(Q_values,actions),reduction_indices=1)
                max_Q_value = tf.reduce_max(Q_values,axis=-1)
                variables = [v for v in tf.trainable_variables() if v.name.startswith('type_selector')]
                variable_summary(variables,['DQN_VAR_SUMM'])
           
            with tf.name_scope('Loss'):
                Q_value_loss = tf.reduce_mean(tf.square(action_Q_value-target_Q_value))
                '''
                #reconstruct_loss = tf.reduce_mean(tf.reduce_sum(tf.square(reconstruct-var_state),1))
                #reconstruct_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(reconstruct-var_state),1))
                if self.var_state_size != 0:
                    DQN_loss = Q_value_loss + 0.1*reconstruct_loss
                else:
                    DQN_loss = Q_value_loss
                '''
                DQN_loss = Q_value_loss
                tf.summary.scalar('Q_value_loss',Q_value_loss,collections=['DQN_SUMM']) 
                #tf.summary.scalar('reconstruct_loss',reconstruct_loss,collections=['DQN_SUMM']) 
                tf.summary.scalar('DQN_loss',DQN_loss,collections=['DQN_SUMM']) 
        
        with tf.variable_scope('Episode', initializer=tf.contrib.layers.xavier_initializer()):
            self.QA_local_step = tf.Variable(0,name='QA_local_step',trainable=False)
            self.QA_local_step_reset = self.QA_local_step.assign(0)
            self.QA_current_global_step = self.QA_global_step + self.QA_local_step
            self.QA_global_step_update = tf.assign(self.QA_global_step,self.QA_global_step+self.QA_local_step)
          
            self.epsilon = tf.placeholder(tf.float32,shape=(),name='epsilon')
            self.temperature = tf.placeholder(tf.float32,shape=(),name='temperature')
            self.final_acc = tf.placeholder(tf.float32,shape=(),name='final_acc')
            tf.summary.scalar('epsilon',self.epsilon,collections=['EPISODE_SUMM'])
            tf.summary.scalar('temperature',self.temperature,collections=['EPISODE_SUMM'])
            tf.summary.scalar('final_acc',self.final_acc,collections=['EPISODE_SUMM'])
            
            self.num_action = tf.placeholder(tf.int32,shape=(),name='num_action')
            self.ave_reward = tf.placeholder(tf.float32,shape=(),name='ave_reward')
            tf.summary.scalar('number_of_action',self.num_action,collections=['TASK_EPISODE_SUMM'])
            tf.summary.scalar('ave_reward',self.ave_reward,collections=['TASK_EPISODE_SUMM'])

        
        with tf.variable_scope('Action', initializer=tf.contrib.layers.xavier_initializer()):
            action_global_step = tf.Variable(0,name='action_global_step',trainable=False)
            self.action_step_add_op = tf.assign(action_global_step,action_global_step+1)
            self.action_global_step = action_global_step
             
            self.selected_action_q = tf.placeholder(tf.float32,shape=(),name='selected_action_q')
            self.reward = tf.placeholder(tf.float32,shape=(),name='reward')
            tf.summary.scalar('selected_action_Q_value',self.selected_action_q,collections=['ACTION_SUMM']) 
            tf.summary.scalar('reward',self.reward,collections=['ACTION_SUMM'])
            
            self.action_q = tf.placeholder(tf.float32,shape=(),name='action_q')
            tf.summary.scalar('action_Q_value',self.action_q,collections=['TASK_ACTION_SUMM'])
        
        # placeholders
        self.x = input
        self.q = question
        self.y = answer
        self.fc = fact_counts
        self.feed_previous = feed_previous

        # type selector placeholders
        self.var_state = var_state
        self.other_state = other_state
        self.actions = actions
        self.target_Q_value = target_Q_value

        # type selector output tensors
        self.Q_values = Q_values
        self.max_Q_value = max_Q_value
        self.DQN_loss = DQN_loss 
        
        # variables of QA
        self.all_QA_vars = all_QA_vars
        self.QA_state_vars = QA_state_vars
        self.QA_state_var_num = QA_state_var_num
        
        # QA output tensors
        self.QA_ans_logits = QA_ans_logits
        self.QA_ans = QA_ans
        self.QA_total_loss = QA_total_loss
        self.num_corrects = num_corrects
        self.QA_accuracy = QA_accuracy
        self.QA_num_corrects = num_corrects
        # optimizer ops
        l_rate = self.params.learning_rate
        self.QA_opt_op = create_opt('QA_opt', self.QA_total_loss, l_rate, self.QA_local_step,clip=5.)
        #if not forward_only:
        rl_l_rate = self.params.rl_learning_rate
        self.DQN_opt_op = create_opt('DQN_opt',self.DQN_loss,rl_l_rate,self.DQN_global_step,clip=10.)

        # merged summary ops
        self.merged_QA = tf.summary.merge_all(key='QA_SUMM')
        self.merged_QA_VAR = tf.summary.merge_all(key='QA_VAR_SUMM')
        self.merged_DQN = tf.summary.merge_all(key='DQN_SUMM')
        self.merged_DQN_VAR = tf.summary.merge_all(key='DQN_VAR_SUMM')
        self.merged_ACTION = tf.summary.merge_all(key='ACTION_SUMM')
        self.merged_TASK_ACTION = tf.summary.merge_all(key='TASK_ACTION_SUMM')
        self.merged_EPISODE = tf.summary.merge_all(key='EPISODE_SUMM')
        self.merged_TASK_EPISODE = tf.summary.merge_all(key='TASK_EPISODE_SUMM')
    '''
    def train_baseline(self,train_data,val_data):
        params = self.params
        assert self.action is not 'test'
        max_action = 1000
        episode = 500
        memory_size = 10000 
        sample_num = 1
        ## QA training memory, should be reset every episoode
        QA_x_mem = QuestionMemory((params.story_size, params.sentence_size), max_action*sample_num, dtype='int32')
        QA_q_mem = QuestionMemory((params.question_size,), max_action*sample_num, dtype='int32')
        QA_y_mem = QuestionMemory((), max_action*sample_num, dtype='int32')
       
        for episo in range(episode):
            ## reset QA memory
            QA_x_mem.reset()
            QA_q_mem.reset()
            QA_y_mem.reset()
            
            action_record = np.zeros((params.action_num))
            self.acc = 0
            for it in range(max_action+1):
                print ('epsiode %d: action %d'%(episo,it),end='\r')
                
                ## decide action
                if it == max_action:
                    action = params.action_num
                else:
                    action = random.randint(0,params.action_num-1)
                                                
                action_record[action] += 1
                ## sample questions from data set
                if action != params.action_num:
                    contexts,questions,ans = train_data[action].get_random_cnt(sample_num)
                    QA_x_mem.append(contexts)
                    QA_q_mem.append(questions)
                    QA_y_mem.append(ans)
                else:         
                    ## get reward
                    reward,converge_time = self.get_DQN_reward(action,val_data,QA_x_mem,QA_q_mem,QA_y_mem)
                    print ('episode %d terminate, acc=%f'%(episo,self.acc))

            ## write episode summary
            feed_dict = {self.final_acc:    self.acc,
                         self.episode_step: episo,
                         self.epsilon:      1}
            for index,tensor in enumerate(self.selected_task_num):
                feed_dict[tensor] = action_record[index]
            episode_summ,episode_global_step = self.sess.run(self.episode_summ_list,feed_dict = feed_dict)
            self.summary_writer.add_summary(episode_summ, episode_global_step)
            
            ## reset QA model
            self.sess.run([self.QA_var_reset,self.QA_global_step_update])
            self.sess.run([self.QA_local_step_reset])
            tf.reset_default_graph() 
            #gc.collect()
    ''' 
    def initial_environment(self,max_action,sample_num,memory_size):
        params = self.params
       
        if not self.initial :
            self.sess.run(self.action_global_step.assign(0))
            ## QA training memory, should be reset every episoode
            self.QA_x_mem = QuestionMemory((params.story_size, params.sentence_size), max_action*sample_num, dtype='int32')
            self.QA_q_mem = QuestionMemory((params.question_size,), max_action*sample_num, dtype='int32')
            self.QA_y_mem = QuestionMemory((), max_action*sample_num, dtype='int32')
            
            self.tmp_QA_x_mem = QuestionMemory((params.story_size, params.sentence_size), sample_num, dtype='int32')
            self.tmp_QA_q_mem = QuestionMemory((params.question_size,), sample_num, dtype='int32')
            self.tmp_QA_y_mem = QuestionMemory((), sample_num, dtype='int32')
            ## DQN memory pool
            self.dqn_memory_pool = DQNMemory([(self.var_state_size,),(self.other_state_size,)],params.action_num,memory_size)

            self.initial = True
    
    def environment_reset(self):
        params = self.params
        ## reset QA memory
        self.QA_x_mem.reset()
        self.QA_q_mem.reset()
        self.QA_y_mem.reset()

        ## reset tmp QA memory
        self.tmp_QA_x_mem.reset()
        self.tmp_QA_q_mem.reset()
        self.tmp_QA_y_mem.reset()
        
        ## initial QA variable
        self.QA_reset()
        
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.action_record = np.zeros((params.action_num))
        self.reward_record = np.zeros((params.action_num))
        self.not_improve = np.zeros((params.action_num))

    def train(self, train_data,val_data,learning_mode):    
        params = self.params 
        assert self.action is not 'test'
        max_action = 5
        episode = 1000
        memory_size = 100000 
        #sample_num = params.batch_size
        sample_num = 50

        ## initial all memory
        self.initial_environment(max_action,sample_num,memory_size) 
        epsilon=0
        T=0
        try:
            for episo in range(episode):
                ## reset environment
                
                if learning_mode == 'epsilon_greedy':
                    ## epsilon greedy
                    epsilon = 1/(1+episo/200)
                elif learning_mode == 'soft_q':
                    ## temperature
                    T = max(100*0.99**(episo//50),0.1)
                
                if (episo+1)%10 == 0:
                    print ('Train DQN on all memory')
                    for _ in range(5):
                        self.DQN_train(num_batch = 10)
                    self.test(train_data,val_data,max_action,sample_num,episo,learning_mode)
                
                self.environment_reset()
                _,self.acc = self.eval(val_data,name='QA')
                print ('episode %d========================'%episo)
                print ('initial acc : %f'%self.acc)
                for it in range(max_action):
                    print ('    action %d'%(it),end='')
                    ## get learner QA variables value(learner state)
                    learner_state = self.get_learner_state(self.state_mode)
                    
                    ## push observation into DQN memory pool
                    self.dqn_pool_add(learner_state)

                    ## get action and Q values
                    action,Q_values = self.decide_action(learner_state,learning_mode,epsilon,T,False)
                    print (', choose task %d'%(action+1),end='')
                    self.action_record[action] += 1
                    
                    ## sample questions from data set
                    if action != params.action_num:
                        contexts,questions,ans = train_data[action].get_batch_cnt(sample_num)
                        self.QA_x_mem.append(contexts)
                        self.QA_q_mem.append(questions)
                        self.QA_y_mem.append(ans)
                        self.tmp_QA_x_mem.append(contexts)
                        self.tmp_QA_q_mem.append(questions)
                        self.tmp_QA_y_mem.append(ans)
                    
                    ## get reward
                    if it == max_action-1 or action == params.action_num:
                        terminate = 0
                    else:
                        terminate = 1
                    reward,converge_time = self.get_DQN_reward(action,val_data,terminate)
                    
                    self.reward_record[action] += reward               
                    print (', reward %f'%(reward),end='')
                    print (', acc %f'%(self.acc))
                    print ('    converge after %d times training'%converge_time)
                    self.write_action_summary('train',Q_values,action,reward)
                    
                    ## memory replay
                    if (self.dqn_memory_pool.size > 5*max_action or episo > 20) and self.dqn_memory_pool.size%10 == 0:
                        self.DQN_train(num_batch = 1)

                    if terminate == 0:
                        break

                    self.last_state = learner_state
                    self.last_action = action
                    self.last_reward = reward 
                    self.last_terminate = terminate
                
                assert terminate == 0    
                self.dqn_memory_pool.append(learner_state,action,reward,learner_state,terminate)
                print ('episode %d terminate, acc=%f'%(episo,self.acc))
                print (self.action_record)
                
                ## write episode summary
                if learning_mode == 'epsilon_greedy':
                    self.write_episode_summary('train',learning_mode,epsilon,episo)
                elif learning_mode == 'soft_q': 
                    self.write_episode_summary('train',learning_mode,T,episo)

                self.sess.run([self.QA_global_step_update])
                self.sess.run([self.QA_local_step_reset])
                tf.reset_default_graph() 
                gc.collect()
                for data_set in train_data:
                    data_set.reset()
        
        except KeyboardInterrupt:
            print ('KeyboardInterrupt')
            self.save_params()
        finally:
            #self.save_dqn_memory(dqn_memory_pool)
            self.save(episo)
    
    def test(self,train_data,val_data,max_action,sample_num,episo,learning_mode):
        params = self.params
        self.initial_environment(max_action,sample_num,0) 
        self.environment_reset()
        
        T = 0.05
        _,self.acc = self.eval(val_data,name='QA')
        print ('testing =========================[ episode %i ]'%episo)
        print ('initial acc : %f'%self.acc)
        for it in range(max_action):
            print ('=== [ action %i ] ==='%it)
            ## get learner QA variables value(learner state)
            learner_state = self.get_learner_state(self.state_mode)

            feed_list  = [self.var_state_summary,self.action_global_step]
            feed_dict = {self.var_state:np.expand_dims(learner_state[0],axis=0)}
            var_state_histogram, action_step = self.sess.run(feed_list,feed_dict=feed_dict)
            self.test_summary_writer.add_summary(var_state_histogram,action_step)
    
            ## decide action
            action,Q_values = self.decide_action(learner_state,learning_mode,0,T,True)
            print ('choose task %d'%(action+1),end='')
            self.action_record[action] += 1
           
            ## sample questions from data set
            if action != params.action_num:
                contexts,questions,ans = train_data[action].get_batch_cnt(sample_num)
                self.QA_x_mem.append(contexts)
                self.QA_q_mem.append(questions)
                self.QA_y_mem.append(ans)
                
                self.tmp_QA_x_mem.append(contexts)
                self.tmp_QA_q_mem.append(questions)
                self.tmp_QA_y_mem.append(ans)

            ## get reward
            if action == params.action_num or it == max_action-1:
                terminate = 0
            else:        
                terminate= 1
            
            reward,converge_time = self.get_DQN_reward(action,val_data,terminate)
            
            self.reward_record[action] += reward
            print (', reward %f'%(reward),end='')
            print (', acc %f'%(self.acc))
            print ('converge after %d times training'%converge_time)
            self.write_action_summary('test',Q_values,action,reward)

           
            self.last_action = action
            self.last_reward = reward
            self.last_state = learner_state
            if terminate == 0:
                break
        assert terminate == 0
        print ('testing, acc=%f'%(self.acc))
        print (self.action_record)
        
        print ('testing end======================')
        ## write episode summary
        if learning_mode == 'epsilon_greedy':
            self.write_episode_summary('test',learning_mode,0,episo)
        elif learning_mode == 'soft_q':
            self.write_episode_summary('test',learning_mode,T,episo)
        elif learning_mode == 'random':
            self.write_episode_summary('test',learning_mode,0,episo)
        
        ## reset QA model
        self.sess.run([self.QA_var_reset,self.QA_global_step_update])
        self.sess.run([self.QA_local_step_reset])
        tf.reset_default_graph() 
        gc.collect()
        for data_set in train_data:
            data_set.reset()

    def dqn_pool_add(self,new_state):    
        if self.last_state is not None and self.last_action is not None:
            self.dqn_memory_pool.append(self.last_state,self.last_action,self.last_reward,new_state,self.last_terminate)
    
    def DQN_train(self,num_batch):
        params = self.params
        dqn_memory_pool = self.dqn_memory_pool
        assert not self.action == 'test'
        discount = 0.8
        size = dqn_memory_pool.size
        batch_size = params.batch_size
        if num_batch == 'all' or size < num_batch * batch_size:
            num_batch = math.ceil(size/batch_size)
        
        index = np.arange(size)
        np.random.shuffle(index)
        for i in range(num_batch):
            if (i+1)*batch_size < size:
                chosen_memory = index[i*batch_size:(i+1)*batch_size]
            else:
                chosen_memory = index[i*batch_size:size]

            states,one_hot_actions,rewards,next_states,terminates = dqn_memory_pool[chosen_memory]
            
            ## get max Q value of next state
            max_Q_value = self.sess.run(self.max_Q_value,feed_dict = {self.var_state:next_states[0],
                                                                      self.other_state:next_states[1]}) # [N]
            
            ## get target value Q(s,a) <- Q(s,a) + lr*( r + discount * max Q(s',a') - Q(s,a))
            #target_Q_value = (1-lr)*Q_value_action + lr*(rewards + discount*max_Q_value * terminates)
            target_Q_value = rewards + discount*max_Q_value * terminates
            DQN_summ,DQN_VAR_summ,DQN_global_step,_ = self.sess.run(self.DQN_train_list,
                                                       feed_dict = {self.var_state:states[0],
                                                                    self.other_state:states[1],
                                                                    self.actions:one_hot_actions,
                                                                    self.target_Q_value:target_Q_value})
            self.summary_writer.add_summary(DQN_summ, DQN_global_step)
            self.var_summary_writer.add_summary(DQN_VAR_summ, DQN_global_step)
    
    def get_learner_state(self,state_mode): 
        params = self.params 
        
        var_state = np.zeros(0,'float32')
        other_state = np.zeros(0,'float32')
        
        if 'QA_var' in state_mode:
            QA_state_vars_value = self.sess.run(self.QA_state_vars)
            for value in QA_state_vars_value:
                var_state = np.append(var_state,value)
        
        if 'action_record' in state_mode:
            other_state = np.append(other_state,self.action_record)             

        if 'last_action' in state_mode:
            one_hot_last_action = np.zeros(params.action_num)
            if last_action is not None:
                one_hot_last_action[self.last_action]=1
            other_state = np.append(other_state,one_hot_last_action)

        if 'last_reward' in state_mode:
            if self.last_reward is not None:
                other_state = np.append(other_state,self.last_reward)
            else:
                other_state = np.append(other_state,0)
        if 'not_improve' in state_mode :
            if self.last_reward == 0:
                self.not_improve[self.last_action] += 1
            else :
                self.not_improve = np.zeros((params.action_num))
            other_state = np.append(other_state,self.not_improve)
            #print (self.not_improve)

        if 'acc' in state_mode :
            other_state = np.append(other_state,self.acc)
        
        return [var_state,other_state]

    def decide_action(self,state,learning_mode,epsilon,T,verbose=False):
        params = self.params
        ## get action Q value 
        Q_values = self.sess.run(self.Q_values,feed_dict = {self.var_state:np.expand_dims(state[0],axis=0),
                                                            self.other_state:np.expand_dims(state[1],axis=0)})
        Q_values = Q_values[0]
        
        if verbose :
            print ('Q_value : ',Q_values)                                
        ## decide action
        if learning_mode == 'epsilon_greedy':
            if random.random() < epsilon : 
                action = random.randint(0,params.action_num)
            else:
                action = np.argmax(Q_values)
        elif learning_mode == 'soft_q': 
            q = Q_values/T
            q -= np.max(q)
            p = np.exp(q)
            p /= np.sum(p)
            action = np.random.choice(params.action_num,1,p=p)[0]
            if verbose:
                print ('probability : ',p)
        elif learning_mode == 'random':
            action = np.random.choice(params.action_num)
        else:
            raise Exception('unknown learning mode %s'%learning_mode)

        return action,Q_values

    def QA_reset(self):
        self.sess.run([self.QA_var_reset])
    
    def get_DQN_reward(self,action,val_data,terminate):
        params = self.params

        converge_time = 0
        tolerance = 10

        if terminate == 0:
            QA_x_mem = self.QA_x_mem
            QA_q_mem = self.QA_q_mem
            QA_y_mem = self.QA_y_mem
        elif terminate == 1:
            QA_x_mem = self.QA_x_mem
            QA_q_mem = self.QA_q_mem
            QA_y_mem = self.QA_y_mem
            '''
            QA_x_mem = self.tmp_QA_x_mem
            QA_q_mem = self.tmp_QA_q_mem
            QA_y_mem = self.tmp_QA_y_mem
            '''


        '''
        if action == params.action_num and len(QA_x_mem) > 0:
            ## train QA until converge
            max_QA_acc = self.acc
            count = 0
            while count < tolerance:
                QA_global_step = self.QA_train(QA_x_mem,QA_q_mem,QA_y_mem,num_batch = 'all')
                converge_time += 1
                _,QA_acc = self.eval(val_data,name='QA')
            
                if QA_acc > max_QA_acc:
                    max_QA_acc = QA_acc
                    count = 0
                else:
                    count += 1
        else :
            _,max_QA_acc = self.eval(val_data,name='QA')
        '''
        max_QA_acc = self.acc
        if len(QA_x_mem) > 0:
            ## train QA until converge
            count = 0
            QA_vars = self.sess.run(self.all_QA_vars)
            while count < tolerance:
                QA_global_step = self.QA_train(QA_x_mem,QA_q_mem,QA_y_mem,num_batch = 'all')
                converge_time += 1
                _,QA_acc = self.eval(val_data,name='QA')
            
                if QA_acc > max_QA_acc:
                    max_QA_acc = QA_acc
                    count = 0
                    QA_vars = self.sess.run(self.all_QA_vars)
                else:
                    count += 1
            for index,var in enumerate(self.all_QA_vars):
                self.sess.run(tf.assign(var,QA_vars[index]))
        #_,max_QA_acc = self.eval(val_data,name='QA')

        #if action == params.action_num:
        #    reward = max_QA_acc
        #else :
        #    reward =0
        reward = max_QA_acc - self.acc
        self.acc = max_QA_acc
    
        '''
        if action == self.last_action and reward < 1e-8:
            reward = -0.1

        reward += 0.05*(1 if action != params.action_num and self.action_record[action] == 1 else 0)
        '''
        return reward, max(converge_time-tolerance,0)
    
    def write_action_summary(self,name,Q_values,action,reward):
        params = self.params
        ## get action global step and selected action q value
        action_summ,action_global_step,_ = self.sess.run(self.action_summ_list,feed_dict={self.selected_action_q:Q_values[action],
                                                                                          self.reward:reward})
        if name == 'train':
            self.summary_writer.add_summary(action_summ,action_global_step) 
        elif name == 'test':
            self.test_summary_writer.add_summary(action_summ,action_global_step) 
        
        ## write Q value of each action(including terminate action)
        for i in range(params.action_num):
            task_action_summ = self.sess.run(self.task_action_summ_list,feed_dict={self.action_q:Q_values[i]})
            self.task_summary_writers[name][i].add_summary(task_action_summ,action_global_step)
    
    def write_episode_summary(self,name,learning_mode,explore_param,episode_step):
        params = self.params
        ## write epsilon and final accuracy
        if learning_mode == 'epsilon_greedy':
            feed_dict = {self.epsilon:explore_param,self.final_acc:self.acc,self.temperature:0}
        elif learning_mode == 'soft_q':
            feed_dict = {self.temperature:explore_param,self.final_acc:self.acc,self.epsilon:0}
        elif learning_mode == 'random':
            feed_dict = {self.temperature:0,self.final_acc:self.acc,self.epsilon:0}
        episode_summ = self.sess.run(self.episode_summ_list,feed_dict=feed_dict)
        
        if name == 'train':
            self.summary_writer.add_summary(episode_summ,episode_step) 
        elif name == 'test':
            self.test_summary_writer.add_summary(episode_summ,episode_step) 
        ## write number of each action and average reward of each action
        for i in range(params.action_num):
            num_action = self.action_record[i]
            ave_reward = self.reward_record[i]/(self.action_record[i]+1e-9)
            task_episode_summ = self.sess.run(self.task_episode_summ_list,feed_dict={self.num_action:num_action,
                                                                                     self.ave_reward:ave_reward})
            self.task_summary_writers[name][i].add_summary(task_episode_summ,episode_step)

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
        self.QA_train_list  = [self.merged_QA, self.merged_QA_VAR,self.QA_current_global_step, self.QA_opt_op]
        #self.QA_train_list  = [self.merged_QA, self.QA_global_step, self.QA_opt_op]
        self.QA_test_list   = [self.merged_QA, self.reQA_global_step+self.QA_global_step+1000, self.QA_total_loss, self.QA_accuracy,self.QA_num_corrects]
        #self.rl_train_list  = [self.merged_RL, self.RL_global_step, self.RL_opt_op]
        #self.D_train_list   = [self.merged_D, self.D_global_step, self.D_opt_op]

        #self.pre_test_list  = [self.merged_QA, self.merged_QG, self.Pre_global_step,
        #                       0.5*self.QA_total_loss+0.5*self.QG_total_loss]
        #self.rl_test_list   = [self.merged_QA, self.merged_QG, self.QA_global_step, self.QA_total_loss]

        self.DQN_train_list = [self.merged_DQN,self.merged_DQN_VAR,self.DQN_global_step,self.DQN_opt_op]
        self.action_summ_list = [self.merged_ACTION,self.action_global_step,self.action_step_add_op]
        self.task_action_summ_list = self.merged_TASK_ACTION
        
        self.episode_summ_list = self.merged_EPISODE
        self.task_episode_summ_list = self.merged_TASK_EPISODE
        self.QA_reset_list = [self.QA_var_reset,self.QA_global_step_update,self.QA_local_step_reset]
    def get_feed_dict(self, batches, feed_previous, is_train, is_sample):
        return {
            self.x: batches[0],
            self.q: batches[1],
            self.y: batches[2],
            self.is_training: is_train,
            self.is_sample: is_sample,
            self.feed_previous: feed_previous
        }

    def save_dqn_memory(self,dqn_memory):
        assert not self.action == 'test'
        params = self.params
        filename = os.path.join(self.save_dir, "dqn_memory.pickle")
        with open(filename, 'wb') as f:
            pickle.dump(dqn_memory, f)
    
    def load_dqn_memory(self,path):
        with open(path,'rb') as file:
            dqn_memory = pickle.load(file)
        return dqn_memory
    
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

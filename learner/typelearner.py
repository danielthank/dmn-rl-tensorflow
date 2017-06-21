import os
import json
import random
import numpy as np
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
        if not self.action == 'test':
            self.task_summary_writers = {'train':[],'test':[]}
            for i in range(params.action_num+1):
                if i != params.action_num :
                    task_summary_dir = os.path.join(self.save_dir,'task_%d'%(i+1))
                    test_task_summary_dir = os.path.join(self.save_dir,'test_task_%d'%(i+1))
                else:
                    task_summary_dir = os.path.join(self.save_dir,'terminate')
                    test_task_summary_dir = os.path.join(self.save_dir,'test_terminate')
                self.task_summary_writers['train'].append(tf.summary.FileWriter(logdir=task_summary_dir,graph=self.sess.graph))
                self.task_summary_writers['test'].append(tf.summary.FileWriter(logdir=test_task_summary_dir,graph=self.sess.graph))
            
            self.summary_dir = os.path.join(self.save_dir, 'train_summary')
            self.validation_summary_dir = os.path.join(self.save_dir, 'train_validation_summary')
            self.var_summary_dir = os.path.join(self.save_dir, 'train_var_summary')
            self.summary_writer = tf.summary.FileWriter(logdir=self.summary_dir,graph=self.sess.graph)
            self.validation_summary_writer = tf.summary.FileWriter(logdir=self.validation_summary_dir,
                                                                   graph=self.sess.graph)
            self.var_summary_writer = tf.summary.FileWriter(logdir=self.var_summary_dir, graph=self.sess.graph)

        ## reward baseline ##
        self.baseline = 0.

        ## define session run lists ##
        self.def_run_list()

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
            
        
        with tf.variable_scope('type_selector', initializer=tf.contrib.layers.xavier_initializer()):
            all_QA_vars = [x for x in tf.trainable_variables() if x.name.startswith('QA')]
            QA_decoder_vars = [x for x in tf.trainable_variables() if x.name.startswith('QA')]
            for var in all_QA_vars:
                print ('name: ',var.name)
                print ('shape: ',var.shape)
            #sys.exit()
            QA_decoder_var_num = sum([reduce((lambda x,y:x*y),var.shape.as_list()) for var in QA_decoder_vars])
            print ('QA_var_num:%d'%QA_decoder_var_num)
            self.QA_var_reset = tf.variables_initializer(all_QA_vars)

            with tf.name_scope('DQN') :
                state = tf.placeholder(tf.float32,shape=[None,QA_decoder_var_num],name='leaner_state')
                actions = tf.placeholder(tf.float32,shape=[None,params.action_num+1],name='learner_actions')
                target_Q_value = tf.placeholder(tf.float32,shape=[None],name='target_Q_values')
                
                fc = tf.layers.dense(state,1024)
                fc = tf.nn.relu(fc)
                fc = tf.layers.dense(fc,128)
                fc = tf.nn.relu(fc)
                Q_values = tf.layers.dense(fc,params.action_num+1) 

                action_Q_value = tf.reduce_sum(tf.multiply(Q_values,actions),reduction_indices=1)
                max_Q_value = tf.reduce_max(Q_values,axis=-1)
           
            with tf.name_scope('Loss'):
                DQN_loss = tf.reduce_mean(tf.square(action_Q_value-target_Q_value))
            tf.summary.scalar('DQN_loss',DQN_loss,collections=['DQN_SUMM']) 
        
        with tf.variable_scope('Episode', initializer=tf.contrib.layers.xavier_initializer()):
            self.QA_local_step = tf.Variable(0,name='QA_local_step',trainable=False)
            self.QA_local_step_reset = self.QA_local_step.assign(0)
            self.QA_current_global_step = self.QA_global_step + self.QA_local_step
            self.QA_global_step_update = tf.assign(self.QA_global_step,self.QA_global_step+self.QA_local_step)
          
            self.epsilon = tf.placeholder(tf.float32,shape=(),name='epsilon')
            self.final_acc = tf.placeholder(tf.float32,shape=(),name='final_acc')
            tf.summary.scalar('epsilon',self.epsilon,collections=['EPISODE_SUMM'])
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
            self.QA_opt_op = create_opt('QA_opt', self.QA_total_loss, l_rate, self.QA_local_step)
            self.DQN_opt_op = create_opt('DQN_opt',self.DQN_loss,l_rate,self.DQN_global_step,clip=10.)

        # merged summary ops
        self.merged_QA = tf.summary.merge_all(key='QA_SUMM')
        self.merged_VAR = tf.summary.merge_all(key='VAR_SUMM')
        self.merged_DQN = tf.summary.merge_all(key='DQN_SUMM')
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
            
            action_record = np.zeros((params.action_num+1))
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
                    reward = self.get_DQN_reward(action,val_data,QA_x_mem,QA_q_mem,QA_y_mem)
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
    def train(self, train_data,val_data):    
        params = self.params
        assert self.action is not 'test'
        max_action = 3
        episode = 500
        memory_size = 10000 
        sample_num = 256
        ## QA training memory, should be reset every episoode
        QA_x_mem = QuestionMemory((params.story_size, params.sentence_size), max_action*sample_num, dtype='int32')
        QA_q_mem = QuestionMemory((params.question_size,), max_action*sample_num, dtype='int32')
        QA_y_mem = QuestionMemory((), max_action*sample_num, dtype='int32')
        
        tmp_QA_x_mem = QuestionMemory((params.story_size, params.sentence_size), sample_num, dtype='int32')
        tmp_QA_q_mem = QuestionMemory((params.question_size,), sample_num, dtype='int32')
        tmp_QA_y_mem = QuestionMemory((), sample_num, dtype='int32')
        ## DQN memory pool
        dqn_memory_pool = DQNMemory((self.QA_decoder_var_num,),params.action_num+1,memory_size)
        try:
            for episo in range(episode):
                ## epsilon greedy
                epsilon = 1/(1+episo/200)
                if (episo+1)%10 == 0:
                    self.test(train_data,val_data,max_action,sample_num,episo)
                ## reset QA memory
                QA_x_mem.reset()
                QA_q_mem.reset()
                QA_y_mem.reset()
                
                last_state = None
                last_action = None
                last_reward = None
                action_record = np.zeros((params.action_num+1))
                reward_record = np.zeros((params.action_num+1))
                self.acc = 0
                for it in range(max_action+1):
                    print ('epsiode %d: action %d'%(episo,it),end='\r')
                    ## get learner QA variables value(learner state)
                    QA_decoder_vars_value = self.sess.run(self.QA_decoder_vars)
                    learner_state = np.zeros(0,'float32')
                    for value in QA_decoder_vars_value:
                        learner_state = np.append(learner_state,value)
                    
                    ## get action Q value 
                    Q_values = self.sess.run(self.Q_values,feed_dict = {self.state:np.expand_dims(learner_state,axis=0)})
                    Q_values = Q_values[0] 
                    
                    ## decide action
                    if it == max_action:
                        action = params.action_num
                    elif random.random() < epsilon: 
                        action = random.randint(0,params.action_num-1)
                    else:
                        action = np.argmax(Q_values[:-1])
                                                    
                    action_record[action] += 1
                    ## sample questions from data set
                    if action != params.action_num:
                        contexts,questions,ans = train_data[action].get_random_cnt(sample_num)
                        QA_x_mem.append(contexts)
                        QA_q_mem.append(questions)
                        QA_y_mem.append(ans)
                        
                        tmp_QA_x_mem.append(contexts)
                        tmp_QA_q_mem.append(questions)
                        tmp_QA_y_mem.append(ans)

                        ## train QA
                        for _ in range(3):
                            QA_global_step = self.QA_train(tmp_QA_x_mem,tmp_QA_q_mem,tmp_QA_y_mem,num_batch = 'all')
                    
                    ## get reward
                    reward = self.get_DQN_reward(action,val_data,QA_x_mem,QA_q_mem,QA_y_mem)
                    reward_record[action] += reward
                    
                    if action == params.action_num:
                        print ('episode %d terminate, acc=%f'%(episo,self.acc))
                        print (action_record)
                        terminate = 0
                    else:
                        terminate= 1
                        self.write_action_summary('train',Q_values,action,reward)
                    ## push observation into DQN memory pool
                    if last_state is not None and last_action is not None:
                        dqn_memory_pool.append(last_state,last_action,last_reward,learner_state,last_terminate)
                    last_state = learner_state
                    last_action = action
                    last_reward = reward 
                    last_terminate = terminate
                    ## memory replay
                    if dqn_memory_pool.size > 2*max_action and dqn_memory_pool.size > params.batch_size and dqn_memory_pool.total_append_size % 50 == 0:
                        self.DQN_train(dqn_memory_pool,num_batch = 5)
                    if terminate == 0:
                        break
                assert terminate == 0
                dqn_memory_pool.append(learner_state,action,reward,learner_state,terminate)
                
                ## write episode summary
                self.write_episode_summary('train',action_record,reward_record,epsilon,episo)
                
                ## reset QA model
                self.sess.run([self.QA_var_reset,self.QA_global_step_update])
                self.sess.run([self.QA_local_step_reset])
                tf.reset_default_graph() 
                #gc.collect()
        except KeyboardInterrupt:
            print ('KeyboardInterrupt')
            self.save_params()
        finally:
            self.save(episo)
    
    def test(self,train_data,val_data,max_action,sample_num,episo):
        params = self.params
        ## QA training memory, should be reset every episoode
        QA_x_mem = QuestionMemory((params.story_size, params.sentence_size), max_action*sample_num, dtype='int32')
        QA_q_mem = QuestionMemory((params.question_size,), max_action*sample_num, dtype='int32')
        QA_y_mem = QuestionMemory((), max_action*sample_num, dtype='int32')
        
        tmp_QA_x_mem = QuestionMemory((params.story_size, params.sentence_size), sample_num, dtype='int32')
        tmp_QA_q_mem = QuestionMemory((params.question_size,), sample_num, dtype='int32')
        tmp_QA_y_mem = QuestionMemory((), sample_num, dtype='int32')
        
        ## reset QA memory
        QA_x_mem.reset()
        QA_q_mem.reset()
        QA_y_mem.reset()
        
        action_record = np.zeros((params.action_num+1))
        reward_record = np.zeros((params.action_num+1))
        self.acc = 0
        print ('testing =========================')
        for it in range(max_action+1):
            ## get learner QA variables value(learner state)
            QA_decoder_vars_value = self.sess.run(self.QA_decoder_vars)
            learner_state = np.zeros(0,'float32')
            for value in QA_decoder_vars_value:
                learner_state = np.append(learner_state,value)
            
            ## get action Q value 
            Q_values = self.sess.run(self.Q_values,feed_dict = {self.state:np.expand_dims(learner_state,axis=0)})
            Q_values = Q_values[0] 
            
            ## decide action
            if it == max_action:
                action = params.action_num
                print ('action %d, choose terminate')
            else:
                action = np.argmax(Q_values[:-1])
                print ('action %d, choose task %d'%(it,action+1))
                                            
            action_record[action] += 1
            
            ## sample questions from data set
            if action != params.action_num:
                contexts,questions,ans = train_data[action].get_random_cnt(sample_num)
                QA_x_mem.append(contexts)
                QA_q_mem.append(questions)
                QA_y_mem.append(ans)
                
                tmp_QA_x_mem.append(contexts)
                tmp_QA_q_mem.append(questions)
                tmp_QA_y_mem.append(ans)

                ## train QA
                for _ in range(3):
                    QA_global_step = self.QA_train(tmp_QA_x_mem,tmp_QA_q_mem,tmp_QA_y_mem,num_batch = 'all')
            
            ## get reward
            reward = self.get_DQN_reward(action,val_data,QA_x_mem,QA_q_mem,QA_y_mem)
            reward_record[action] += reward
            
            if action == params.action_num:
                print ('testing, acc=%f'%(self.acc))
                print (action_record)
                terminate = 0
            else:
                terminate= 1
                self.write_action_summary('test',Q_values,action,reward)
            
            if terminate == 0:
                break
        assert terminate == 0
        
        print ('testing end======================')
        ## write episode summary
        self.write_episode_summary('test',action_record,reward_record,0,episo)
        
        ## reset QA model
        self.sess.run([self.QA_var_reset,self.QA_global_step_update])
        self.sess.run([self.QA_local_step_reset])
        tf.reset_default_graph() 
        #gc.collect()

        
    def DQN_train(self,dqn_memory_pool,num_batch):
        params = self.params
        assert not self.action == 'test'
        discount = 0.8
        index = np.arange(dqn_memory_pool.size)
        for i in range(num_batch):
            chosen_memory = np.random.choice(index,params.batch_size) 
            states,one_hot_actions,rewards,next_states,terminates = dqn_memory_pool[chosen_memory]
            
            ## get max Q value of next state
            max_Q_value = self.sess.run(self.max_Q_value,feed_dict = {self.state:next_states}) # [N]
            
            ## get target value Q(s,a) <- Q(s,a) + lr*( r + discount * max Q(s',a') - Q(s,a))
            #target_Q_value = (1-lr)*Q_value_action + lr*(rewards + discount*max_Q_value * terminates)
            target_Q_value = rewards + discount*max_Q_value * terminates
            DQN_summ,DQN_global_step,_ = self.sess.run(self.DQN_train_list,
                                                       feed_dict = {self.state:states,
                                                                    self.actions:one_hot_actions,
                                                                    self.target_Q_value:target_Q_value})
            self.summary_writer.add_summary(DQN_summ, DQN_global_step)
   
    def get_DQN_reward(self,action,val_data,QA_x_mem,QA_q_mem,QA_y_mem):
        params = self.params

        if action == params.action_num and len(QA_x_mem) > 0:
            ## train QA until converge
            max_QA_acc = 0
            count = 0
            while count < 5:
                QA_global_step = self.QA_train(QA_x_mem,QA_q_mem,QA_y_mem,num_batch = 'all')
                _,QA_acc = self.eval(val_data,name='QA')
            
                if QA_acc > max_QA_acc:
                    max_QA_acc = QA_acc
                    count = 0
                else:
                    count += 1
        else :
            _,max_QA_acc = self.eval(val_data,name='QA')
        '''
        if len(QA_x_mem) > 0:
            ## train QA until converge
            max_QA_acc = 0
            count = 0
            while count < 5:
                QA_global_step = self.QA_train(QA_x_mem,QA_q_mem,QA_y_mem,num_batch = 'all')
                _,QA_acc = self.eval(val_data,name='QA')
            
                if QA_acc > max_QA_acc:
                    max_QA_acc = QA_acc
                    count = 0
                else:
                    count += 1
        else :
            _,max_QA_acc = self.eval(val_data,name='QA')
        '''
        #_,max_QA_acc = self.eval(val_data,name='QA')

        if action == params.action_num:
            reward = max_QA_acc
        else :
            reward = max_QA_acc - self.acc
        self.acc = max_QA_acc
        
        return reward
    
    def write_action_summary(self,name,Q_values,action,reward):
        params = self.params
        ## get action global step and selected action q value
        action_summ,action_global_step,_ = self.sess.run(self.action_summ_list,feed_dict={self.selected_action_q:Q_values[action],
                                                                                          self.reward:reward})
        self.summary_writer.add_summary(action_summ,action_global_step) 
        
        ## write Q value of each action(including terminate action)
        for i in range(params.action_num+1):
            task_action_summ = self.sess.run(self.task_action_summ_list,feed_dict={self.action_q:Q_values[i]})
            self.task_summary_writers[name][i].add_summary(task_action_summ,action_global_step)
    
    def write_episode_summary(self,name,action_record,reward_record,epsilon,episode_step):
        params = self.params
        ## write epsilon and final accuracy
        episode_summ = self.sess.run(self.episode_summ_list,feed_dict={self.epsilon:epsilon,
                                                                  self.final_acc:self.acc})
        self.summary_writer.add_summary(episode_summ,episode_step) 
        ## write number of each action and average reward of each action
        for i in range(params.action_num+1):
            num_action = action_record[i]
            ave_reward = reward_record[i]/(action_record[i]+1e-9)
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
        self.QA_train_list  = [self.merged_QA, self.QA_current_global_step, self.QA_opt_op]
        #self.QA_train_list  = [self.merged_QA, self.QA_global_step, self.QA_opt_op]
        self.QA_test_list   = [self.merged_QA, self.reQA_global_step+self.QA_global_step+1000, self.QA_total_loss, self.QA_accuracy]
        #self.rl_train_list  = [self.merged_RL, self.RL_global_step, self.RL_opt_op]
        #self.D_train_list   = [self.merged_D, self.D_global_step, self.D_opt_op]

        #self.pre_test_list  = [self.merged_QA, self.merged_QG, self.Pre_global_step,
        #                       0.5*self.QA_total_loss+0.5*self.QG_total_loss]
        #self.rl_test_list   = [self.merged_QA, self.merged_QG, self.QA_global_step, self.QA_total_loss]

        self.DQN_train_list = [self.merged_DQN,self.DQN_global_step,self.DQN_opt_op]
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

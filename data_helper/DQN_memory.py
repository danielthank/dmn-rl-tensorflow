from collections import deque
import warnings
import numpy as np
from question_memory import QuestionMemory

class DQNMemory():
    def __init__(self,state_shape,action_num,max_len):
        self.state_mem = QuestionMemory(state_shape,max_len)
        self.action_mem = QuestionMemory((),max_len)  ## one hot action
        self.reword_mem = QuestionMemory((),max_len)
        self.next_state_mem = QuestionMemory(state_shape,max_len)
        self.terminate_mem = QuestionMemory((),max_len)
        self.size = 0
        self.state_shape = state_shape
        self.action_num = action_num
        self.max_len = max_len
    
    def reset(self): 
        self.state_mem.reset()
        self.action_mem.reset()
        self.reword_mem.reset()
        self.next_state_mem.reset()
        self.terminate_mem.reset()
        self.size = 0
    
    def append(self,state,action,reword,next_state,terminate):
        assert state.shape == self.state_shape
        assert next_state.shape == self.state_shape
        
        self.state_mem.append(np.expand_dims(state,axis=0))
        self.action_mem.append(np.array([[action]]))
        self.reword_mem.append(np.array([[reword]]))
        self.next_state_mem.append(np.expand_dims(next_state,axis=0))
        self.terminate_mem.append(np.array([[terminate]]))
        
        if self.size < self.max_len:
            self.size += 1
    def __getitem__(self,key):
        states = self.state_mem[key]
        
        actions = self,action_mem[key]
        one_hot_actions = np.zeros((actions.shape[0],self.action_num),dtype='int32')
        one_hot_actions[np.arange(actions.shape[0]),actions] = 1
        
        rewords = self.reword_mem[key]
        next_states = self.next_state_mem[key]
        terminates = self.terminate_mem[key]
        return states,one_hot_actions,rewords,next_states,terminates 


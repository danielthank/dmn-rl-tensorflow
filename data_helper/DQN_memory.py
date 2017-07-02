from collections import deque
import warnings
import numpy as np
from .question_memory import QuestionMemory

class DQNMemory():
    def __init__(self,state_shape,action_num,max_len):
        self.size = 0
        self.total_append_size = 0
        self.action_num = action_num
        self.max_len = max_len
        
        ## init memory
        self.states_mem = []
        self.next_states_mem = []
        if type(state_shape) == list:
            self.list_state = True
            self.state_shape = state_shape
            self.state_num = len(state_shape)
            for shape in state_shape:
                self.states_mem.append(QuestionMemory(shape,max_len))
                self.next_states_mem.append(QuestionMemory(shape,max_len))
        else:
            self.list_state = False
            self.state_shape = [state_shape]
            self.state_num = 1
            self.states_mem.append(QuestionMemory(state_shape,max_len))
            self.next_states_mem.append(QuestionMemory(state_shape,max_len))
        self.action_mem = QuestionMemory((),max_len,dtype=np.int32)  
        self.reword_mem = QuestionMemory((),max_len)
        self.terminate_mem = QuestionMemory((),max_len)
        
            
    def reset(self):
        for index in range(self.state_num):
            self.states_mem[index].reset()
            self.next_states_mem[index].reset()
        self.action_mem.reset()
        self.reword_mem.reset()
        self.terminate_mem.reset()
        self.size = 0
    
    def append(self,state,action,reword,next_state,terminate):
        if not self.list_state: 
            state = [state]
            next_state = [next_state]
        else:
            assert type(state) == list and type(next_state) == list

        for index in range(self.state_num):
            assert state[index].shape == self.state_shape[index]
            assert next_state[index].shape == self.state_shape[index]
            self.states_mem[index].append(np.expand_dims(state[index],axis=0))
            self.next_states_mem[index].append(np.expand_dims(next_state[index],axis=0))
        
        self.action_mem.append(np.array([[action]]))
        self.reword_mem.append(np.array([[reword]]))
        self.terminate_mem.append(np.array([[terminate]]))
        
        if self.size < self.max_len:
            self.size += 1
        self.total_append_size += 1
    
    def __getitem__(self,key):
        states = [mem[key] for mem in self.states_mem]
        
        actions = self.action_mem[key]
        one_hot_actions = np.zeros((actions.shape[0],self.action_num),dtype='int32')
        one_hot_actions[np.arange(actions.shape[0]),actions] = 1
        
        rewords = self.reword_mem[key]
        next_states = [mem[key] for mem in self.next_states_mem]
        terminates = self.terminate_mem[key]
        
        if not self.list_state:
            states = states[0]
            next_states = next_states[0]
        return states,one_hot_actions,rewords,next_states,terminates 


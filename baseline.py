#!/usr/bin/env python3
import os
import sys
import json
import tensorflow as tf
import time
import argparse
from collections import namedtuple
from copy import deepcopy
import numpy as np
from data_helper.read_data import read_babi
from data_helper.data_utils import WordTable


#from baseline_model.dmn import DMN as LEARNER_DMN
from expert.seq2seq import Seq2Seq
#from baseline_model.ren import REN as LEARNER_REN

## accessible model ##
'''
MODEL = {'expert_dmn':      EXPERT_DMN,
         'expert_ren':      EXPERT_REN,
         'learner_dmn':     LEARNER_DMN,
         'learner_seq2seq': LEARNER_Seq2Seq,
         'learner_ren':     LEARNER_REN}
'''
MODEL = {'seq2seq' : Seq2Seq}

def load_params_dict(filename):
    with open(filename, 'r') as file:
        params_dict = json.load(file)
    return params_dict


## arguments parser ##
parser = argparse.ArgumentParser(description='Baseline model')

# Action and arch
parser.add_argument('action', choices=['train', 'test'])
parser.add_argument('arch', choices=['dmn', 'seq2seq', 'ren'])

# directory
parser.add_argument('--load_dir', default='')

# training options
parser.add_argument('--task', default='sweep', type=str, choices=[str(i) for i in range(1, 21)].extend(['all','sweep']))
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--num_epochs', default=256, type=int)
parser.add_argument('--learning_rate', default=0.002, type=float)
parser.add_argument('--val_ratio', default=0.1, type=float)
parser.add_argument('--acc_period', default=10, type=int)
parser.add_argument('--val_period', default=40, type=int)

# dmn params
parser.add_argument('--dmn_memory_step', default=3, type=int)
parser.add_argument('--dmn_memory_update', default='relu')
parser.add_argument('--dmn_embedding_size', default=80, type=int)
parser.add_argument('--dmn_weight_decay', default=0.001, type=float)
parser.add_argument('--dmn_keep_prob', default=1., type=float)
parser.add_argument('--dmn_batch_norm', dest='dmn_batch_norm', action='store_true')
parser.add_argument('--no_dmn_batch_norm', dest='dmn_batch_norm', action='store_false')
parser.set_defaults(dmn_batch_norm=True)

# ren params
parser.add_argument('--ren_embedding_size', default=100, type=int)
parser.add_argument('--ren_num_blocks', default=20, type=int)

args = parser.parse_args()


## main function ##
def main(_):
    
    ## import main model ##
    main_model_name = args.arch
    if main_model_name in MODEL:
        MainModel = MODEL[main_model_name]
    else:
        raise Exception("Unsupported target-arch pair!")

    ## create save dir ##
    save_dir = os.path.join('save_baseline', '{}_{}'.format(args.arch, args.task))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if not os.path.exists(save_dir+'/record'):
        os.makedirs(save_dir+'/record', exist_ok=True)
    args.save_dir = save_dir

    if args.task == 'all':
        task_list = [list(range(1,21))]
    elif args.task == 'sweep':
        task_list = [[str(i)] for i in range(1,21)]
    else:
        task_list = [[int(args.task)]]
    
    for task in task_list:
        ## data set ##
        train, test, words, args.story_size, args.sentence_size, args.question_size = read_babi(task, args.batch_size,
                                                                                            False)
        val = train.split_dataset(args.val_ratio)
        print("training count: {}".format(train.count))
        print("testing count: {}".format(test.count))
    
        print("story size: {}".format(args.story_size))
        print("sentence size: {}".format(args.sentence_size))
        print("question size: {}".format(args.question_size))
        
        ## create params ##
        params_dict = vars(args)
        params_class = namedtuple('params_class', params_dict.keys())
        params = params_class(**params_dict)
        
        record = np.array(['training sample','train acc','val acc','test acc'])
        record = np.expand_dims(record,axis=0)
        max_sample = min(train.count+1,2001)
        max_sample = 201
        for training_sample in range(100,max_sample,100):
            tmp = np.zeros(4,dtype='float32')
            tmp[0] = training_sample
            print ('training data sample  : ',training_sample)
            train_sub = train[:training_sample]
            ## run action ##
            main_model = MainModel(words, params)
            main_model.train(train_sub, val)
            #main_model.save_params()

            main_model.eval(test, name='Test')
            #main_model.decode(test, sys.stdout, sys.stdin, all=False)
            tmp[1] = main_model.train_acc
            tmp[2] = main_model.val_acc
            tmp[3] = main_model.test_acc
            tmp = np.expand_dims(tmp,axis = 0).astype('str')
            record = np.concatenate((record,tmp),axis=0)
        if args.task == 'all':
            output_path = save_dir + '/record/all.csv'
        else :
            output_path = save_dir + '/record/%s.csv' % task[0]
        np.savetxt(output_path,record,fmt ='%s,%s,%s,%s' )
        print ('save record to ',output_path)

if __name__ == '__main__':
    tf.app.run()

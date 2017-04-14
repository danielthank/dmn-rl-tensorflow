#!/usr/bin/python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import json
import tensorflow as tf
import time
import argparse
from collections import namedtuple
from copy import deepcopy

from data_helper.read_data import read_babi
from data_helper.data_utils import WordTable

from expert.dmn import DMN as EXPERT_DMN
from expert.ren import REN as EXPERT_REN

from learner.dmn import DMN as LEARNER_DMN
from learner.seq2seq import Seq2Seq as LEARNER_Seq2Seq
from learner.ren import REN as LEARNER_REN

## accessible model ##
MODEL = {'expert_dmn':      EXPERT_DMN,
         'expert_ren':      EXPERT_REN,
         'learner_dmn':     LEARNER_DMN,
         'learner_seq2seq': LEARNER_Seq2Seq,
         'learner_ren':     LEARNER_REN}


def load_params_dict(filename):
    with open(filename, 'r') as file:
        params_dict = json.load(file)
    return params_dict


## arguments parser ##
parser = argparse.ArgumentParser(description='Expert-Learner dmn and ren')

# Action and target and arch
parser.add_argument('action', choices=['train', 'test', 'rl'])
parser.add_argument('target', choices=['expert', 'learner'])
parser.add_argument('arch', choices=['dmn', 'seq2seq', 'ren'])

# directory
parser.add_argument('--expert_dir', default='')
parser.add_argument('--load_dir', default='')

# training options
parser.add_argument('--task', default='1', type=str, choices=[str(i) for i in range(1, 21)].append('all'))
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
parser.add_argument('--dmn_hidden_size', default=80, type=int)
parser.add_argument('--dmn_weight_decay', default=0.001, type=float)
parser.add_argument('--dmn_keep_prob', default=1., type=float)
parser.add_argument('--dmn_batch_norm', dest='dmn_batch_norm', action='store_true')
parser.add_argument('--no_dmn_batch_norm', dest='dmn_batch_norm', action='store_false')
parser.set_defaults(dmn_batch_norm=True)

# seq2seq params
parser.add_argument('--seq2seq_hidden_size', default=150, type=int)
parser.add_argument('--seq2seq_weight_decay', default=0.001, type=float)

# ren params
parser.add_argument('--ren_embedding_size', default=100, type=int)
parser.add_argument('--ren_num_blocks', default=20, type=int)

args = parser.parse_args()


## main function ##
def main(_):
    ## import main model ##
    main_model_name = '{}_{}'.format(args.target, args.arch)
    if main_model_name in MODEL:
        MainModel = MODEL[main_model_name]
    else:
        raise Exception("Unsupported target-arch pair!")

    ## create save dir ##
    save_dir = os.path.join('save', '{}_{}_{}'.format(args.target, args.arch, args.task))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    args.save_dir = save_dir

    ## data set ##
    if args.task == 'all':
        args.task = list(range(1, 21))
    else:
        args.task = [int(args.task)]
    train, test, words, args.story_size, args.sentence_size, args.question_size = read_babi(args.task, args.batch_size,
                                                                                            args.target=='expert')
    val = train.split_dataset(args.val_ratio)
    print("training count: {}".format(train.count))
    print("testing count: {}".format(test.count))
    """
    print("word2idx:", words.word2idx)
    print("idx2word:", words.idx2word)
    print("word2dc:", words.word2dc)
    print("idx2dc:", words.idx2dc)
    """
    print("story size: {}".format(args.story_size))
    print("sentence size: {}".format(args.sentence_size))
    print("question size: {}".format(args.question_size))

    ## create params ##
    params_dict = vars(args)
    params_class = namedtuple('params_class', params_dict.keys())
    params = params_class(**params_dict)

    if not params.load_dir == '':
        params_filename = os.path.join(params.load_dir, 'params.json')
        load_params = load_params_dict(params_filename)
        if not load_params['task'] == params.task:
            raise Exception("incompatible task with load model!")
        if (not load_params['target'] == params.target) or (not load_params['arch'] == params.arch):
            raise Exception("incompatible main model with load model!")
        params = params._replace(**load_params)

    if not params.expert_dir == '':
        params_filename = os.path.join(params.expert_dir, 'params.json')
        load_params = load_params_dict(params_filename)
        if not load_params['task'] == params.task:
            raise Exception("incompatible task with expert model!")
        if not load_params['target'] == 'expert':
            raise Exception("dir contains no expert model!")
        expert_params = params._replace(action='test', load_dir=params.expert_dir, **load_params)
    else:
        if params.target == 'learner':
            raise Exception("Need to load an expert from expert_dir to run a learner!")
        expert_params = None

    ## run action ##
    if args.action == 'train':
        main_model = MainModel(words, params, expert_params)
        main_model.train(train, val)
        main_model.save_params()

    elif args.action == 'test':
        if args.load_dir == '':
            raise Exception("Need a trained model to test!")
        main_model = MainModel(words, params, expert_params)
        main_model.decode(test, sys.stdout, sys.stdin, all=False)
        main_model.eval(test, name='Test')

    elif args.action == 'rl':
        if not args.target == 'learner':
            raise Exception("Only learner can run rl action!")
        main_model = MainModel(words, params, expert_params)
        main_model.rl_train(train, val)
        main_model.save_params()


if __name__ == '__main__':
    tf.app.run()

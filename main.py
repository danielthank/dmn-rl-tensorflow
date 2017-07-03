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
from expert.seq2seq import Seq2Seq as EXPERT_Seq2Seq
from expert.ren import REN as EXPERT_REN

from learner.dmn import DMN as LEARNER_DMN
from learner.seq2seq import Seq2Seq as LEARNER_Seq2Seq
from learner.ren import REN as LEARNER_REN

from lm.rnnlm import RNNLM as LM_RNNLM 
from lm.lm_helper import train_lm, test_lm

from baseline import run_baseline
from experiments import run_experiments


## accessible model ##
MODEL = {'expert_dmn':      EXPERT_DMN,
         'expert_seq2seq':  EXPERT_Seq2Seq,
         'expert_ren':      EXPERT_REN,
         'learner_dmn':     LEARNER_DMN,
         'learner_seq2seq': LEARNER_Seq2Seq,
         'lm_RNNLM':        LM_RNNLM,
         'learner_ren':     LEARNER_REN}


def load_params_dict(filename):
    with open(filename, 'r') as file:
        params_dict = json.load(file)
    return params_dict


def train_normal(model, params, expert_params, lm_params, words, train, val):
    main_model = model(words, params, expert_params, lm_params)
    main_model.pre_train(train, val)
    main_model.save_params()


def test_normal(model, params, expert_params, lm_params, words, test):
    main_model = model(words, params, expert_params, lm_params)
    main_model.decode(test, sys.stdout, sys.stdin, all=False)
    main_model.eval(test, name='Test')


## arguments parser ##
parser = argparse.ArgumentParser(description='Expert-Learner dmn and ren')

# Action and target and arch
parser.add_argument('action', choices=['train', 'test', 'rl', 'baseline', 'experiments_rl', 'experiments_nonrl'])
parser.add_argument('target', choices=['expert', 'learner', 'lm'])
parser.add_argument('arch', choices=['dmn', 'seq2seq', 'ren', 'RNNLM'])

# directory
parser.add_argument('--expert_dir', default='')
parser.add_argument('--load_dir', default='')
parser.add_argument('--lm_dir', default='')

# training options
parser.add_argument('--task', nargs='+', default=['1'], type=str, choices=[str(i) for i in range(1, 21)]+['all'])
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_epochs', default=256, type=int)
parser.add_argument('--learning_rate', default=0.002, type=float)
parser.add_argument('--rl_learning_rate', default=0.001, type=float)
parser.add_argument('--val_ratio', default=0.1, type=float)
parser.add_argument('--acc_period', default=10, type=int)
parser.add_argument('--val_period', default=40, type=int)
parser.add_argument('--gpu_fraction', default=0.8, type=float)

# dmn params
parser.add_argument('--dmn_memory_step', default=3, type=int)
parser.add_argument('--dmn_memory_update', default='relu')
parser.add_argument('--dmn_embedding_size', default=50, type=int)
parser.add_argument('--dmn_hidden_size', default=50, type=int)
parser.add_argument('--dmn_weight_decay', default=0.001, type=float)
parser.add_argument('--dmn_keep_prob', default=1., type=float)
parser.add_argument('--dmn_batch_norm', dest='dmn_batch_norm', action='store_true')
parser.add_argument('--no_dmn_batch_norm', dest='dmn_batch_norm', action='store_false')
parser.set_defaults(dmn_batch_norm=True)

# seq2seq params
parser.add_argument('--seq2seq_hidden_size', default=50, type=int)
parser.add_argument('--seq2seq_weight_decay', default=0.001, type=float)

# ren params
parser.add_argument('--ren_embedding_size', default=20, type=int)
parser.add_argument('--ren_num_blocks', default=10, type=int)

# RNNLM params
parser.add_argument('--rnnlm_layers', default=2, type=int)
parser.add_argument('--rnnlm_hidden_size', default=50, type=int)
parser.add_argument('--rnnlm_keep_prob', default=0.5, type=float)
parser.add_argument('--lm_num_epoch', default=35, type=int)
parser.add_argument('--lm_ptb_path', default='./ptb', type=str)
parser.add_argument('--lm_num_steps', default=20, type=int)
parser.add_argument('--lm_batch_size', default=20, type=int)
parser.add_argument('--lm_vocab_size', default=20, type=int)
parser.add_argument('--lm_period', default=1, type=int)
parser.add_argument('--lm_val_period', default=1, type=int)

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
    if args.action == 'baseline':
        save_dir = os.path.join('save_baseline', '{}_{}'.format(args.arch, '_'.join(args.task)))
        if not os.path.exists(save_dir+'/record'):
            os.makedirs(save_dir+'/record', exist_ok=True)
    elif args.action == 'experiments_rl':
        save_dir = os.path.join('save_experiments_rl', '{}_{}'.format(args.arch, '_'.join(args.task)))
    elif args.action == 'experiments_nonrl':
        save_dir = os.path.join('save_experiments_nonrl', '{}_{}'.format(args.arch, '_'.join(args.task)))
    elif args.target == 'lm':
        save_dir = os.path.join('save', '{}_{}'.format(args.target, args.arch))
    else:
        save_dir = os.path.join('save', '{}_{}_{}'.format(args.target, args.arch, '_'.join(args.task)))
    args.save_dir = save_dir

    ## data set ##
    if 'all' in args.task:
        args.task = list(range(1, 21))
    else:
        args.task = [int(i) for i in args.task]
    train, test, words, args.story_size, args.sentence_size, args.question_size = read_babi(args.task, args.batch_size,
                                                                                            args.target=='expert')
    val = train.split_dataset(args.val_ratio)
    print("training count: {}".format(train.count))
    print("testing count: {}".format(test.count))
    print("word2idx:", len(words.word2idx))
    print("idx2word:", len(words.idx2word))
    print("story size: {}".format(args.story_size))
    print("sentence size: {}".format(args.sentence_size))
    print("question size: {}".format(args.question_size))

    ## create params ##
    params_dict = vars(args)
    params_class = namedtuple('params_class', params_dict.keys())
    params = params_class(**params_dict)

    ## check target-action-dirs ##
    if params.target == 'lm':
        if params.action == 'train':
            if params.load_dir or params.expert_dir or params.lm_dir:
                raise Exception("No dir needed while training %s model!" % params.target)
            train_process = train_lm
        elif params.action == 'test':
            if not params.load_dir:
                raise Exception("Need a trained %s model to test!" % params.target)
            if params.expert_dir or params.lm_dir:
                raise Exception("No other dirs needed while testing %s model!" % params.target)
            test_process = test_lm
        else:
            raise Exception("Unsupported action for %s!" % params.target)
    elif params.target == 'expert':
        if params.action == 'train':
            if params.load_dir or params.expert_dir or params.lm_dir:
                raise Exception("No dir needed while training %s model!" % params.target)
            train_process = train_normal
        elif params.action == 'test':
            if not params.load_dir:
                raise Exception("Need a trained %s model to test!" % params.target)
            if params.expert_dir or params.lm_dir:
                raise Exception("No other dirs needed while testing %s model!" % params.target)
            test_process = test_normal
        else:
            raise Exception("Unsupported action for %s!" % params.target)
    elif params.target == 'learner':
        if params.action == 'train':
            if params.load_dir:
                raise Exception("Learner in %s mode can only be trained from scratch!" % params.action)
            if (not params.expert_dir) or (not params.lm_dir):
                raise Exception("Two auxiliary models are needed in learner %s mode! " % params.action)
            train_process = train_normal
        elif params.action == 'test':
            if (not params.load_dir) or (not params.expert_dir) or (not params.lm_dir):
                raise Exception("All dirs needed while in learner %s mode!" % params.action)
            test_process = test_normal
        elif params.action == 'rl':
            if (not params.load_dir) or (not params.expert_dir) or (not params.lm_dir):
                raise Exception("All dirs needed while in learner %s mode!" % params.action)
        else:
            if params.load_dir:
                raise Exception("Learner in %s mode can only be trained from scratch!" % params.action)
            if (not params.expert_dir) or (not params.lm_dir):
                raise Exception("Two auxiliary models are needed in learner %s mode! " % params.action)

    ## load params from load_dir ##
    if not params.load_dir == '':
        params_filename = os.path.join(params.load_dir, 'params.json')
        load_params = load_params_dict(params_filename)
        if not load_params['task'] == params.task:
            raise Exception("incompatible task with load model!")
        if (not load_params['target'] == params.target) or (not load_params['arch'] == params.arch):
            raise Exception("incompatible main model with load model!")
        params = params._replace(**load_params)
    else:
        if tf.gfile.Exists(save_dir):
            tf.gfile.DeleteRecursively(save_dir)
        os.makedirs(save_dir, exist_ok=True)

    ## load expert params from expert_dir ##
    if not params.expert_dir == '':
        params_filename = os.path.join(params.expert_dir, 'params.json')
        load_params = load_params_dict(params_filename)
        if not load_params['task'] == params.task:
            raise Exception("incompatible task with expert model!")
        if not load_params['target'] == 'expert':
            raise Exception("dir contains no expert model!")
        expert_params = params._replace(action='test', load_dir=params.expert_dir, **load_params)
    else:
        expert_params = None

    ## load lm params from lm_dir ##
    if not params.lm_dir == '':
        params_filename = os.path.join(params.lm_dir, 'params.json')
        load_params = load_params_dict(params_filename)
        if not load_params['target'] == 'lm':
            raise Exception("dir contains no language model!")
        lm_params = params._replace(action='test', 
                                    load_dir=params.lm_dir,
                                    lm_num_steps=1,
                                    lm_batch_size=params.batch_size,
                                    **load_params)
    else:
        lm_params = None

    ## run action ##
    if args.action == 'train':
        train_process(MainModel, params, expert_params, lm_params, words, train, val)

    elif args.action == 'test':
        test_process(MainModel, params, expert_params, lm_params, words, test)

    elif args.action == 'rl':
        main_model = MainModel(words, params, expert_params, lm_params)
        main_model.rl_train(train, val)
        main_model.save_params()

    elif args.action == 'baseline':
        run_baseline(MainModel, params, expert_params, lm_params, words, train, val)

    elif args.action == 'experiments_rl':
        run_experiments(MainModel, params, expert_params, lm_params, words, train, val, RL=True)

    elif args.action == 'experiments_nonrl':
        run_experiments(MainModel, params, expert_params, lm_params, words, train, val, RL=False)


if __name__ == '__main__':
    tf.app.run()

#!/usr/bin/env python3
import os
import sys
import tensorflow as tf
import time
import argparse

from data_helper.read_data import read_babi, get_max_sizes
from data_helper.data_utils import WordTable

parser = argparse.ArgumentParser(description='Expert-Learner dmn and ren')

# Action and arch and task
parser.add_argument('action', choices=['train', 'test', 'rl'])
parser.add_argument('target', choices=['expert', 'learner', 'xxx'])
parser.add_argument('arch', choises=['dmn', 'seq2seq', 'ren', 'xxx'])

# directory
parser.add_argument('--data_dir', default='babi')
parser.add_argument('--expert_dir', default='')
parser.add_argument('--learner_dir', default='')

# training options
parser.add_argument('--task', default=1, type=int, choices=range(1, 21))
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_epochs', default=0.002, type=float)
parser.add_argument('--val_ratio', default=0.1, type=float)
parser.add_argument('--acc_period', default=10, type=int)
parser.add_argument('--val_period', default=40, type=int)
parser.add_argument('--save_period', default=80, type=int)

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
parser.add_argument('--ren_embedding_size', default=80, type=int)
parser.add_argument('--ren_num_blocks', default=20, type=int)

args = parser.parse_args()

def main(_):
    load = (args.target == 'xxx' and args.arch == 'xxx') 
    if args.target == 'expert':
        if load:
            if args.expert_dir == '':
                raise Exception('expert_dir not specified')
            elif not os.path.exist(args.expert_dir):
                raise Exception('expert_dir not exists')
            dir = args.expert_dir
        else:
            dir = os.path.join('save', {}_{}_{}.format(args.target, args.arch, args.task))
            if not os.path.exists(dir):
                os.makedirs(dir, exist_ok=True)
    else if args.target == 'learner':
        if load:
            if args.leaner_dir == '':
                raise Exception('leaner_dir not specified')
            elif not os.path.exist(args.leaner_dir):
                raise Exception('leaner_dir not exists')
            dir = args.learner_dir
        else:
            dir = os.path.join('save', {}_{}_{}.format(args.target, args.arch, args.task))
            if not os.path.exists(dir):
                os.makedirs(dir, exist_ok=True)
    args.load = load;
    args.dir = dir;

    if args.target == 'expert' and args.arch == 'dmn':
        from Q2A.dmn import DMN as MainModel
    elif args.target == 'expert' and args.arch == 'ren':
        from Q2A.ren import REN as MainModel
    elif args.target == 'learner' and args.arch == 'dmn':
        from A2Q.dmn import DMN as MainModel
    elif args.target == 'learner' and args.arch == 'seq2seq':
        from A2Q.seq2seq import Seq2Seq as MainModel
    else:
        raise Exception('Unsupported model!')

    words = WordTable()
    train = read_babi(os.path.join('save', 'train'), args.task, 'train', args.batch_size, words)
    test = read_babi(os.path.join('save', 'test'), args.task, 'test', args.batch_size, words)
    val = train.split_dataset(args.val_ratio)
    args.sentence_size, args.question_size, args.story_size = get_max_sizes(train, test, val)

    if args.action == 'train':
        summary_dir = os.path.join(args.dir, "summary")
        if tf.gfile.Exists(summary_dir):
            tf.gfile.DeleteRecursively(summary_dir)

        model = Model(args, words)
        if args.load: model.load()
        model.train(train, val)
        model.save_flags()

    elif args.action == 'test':
        model = Model(args, words)
        model.load()
        model.eval(test, name='Test')
        model.decode(test, sys.stdout, all=False)
    
    elif args.action == 'rl':
        pass

if __name__ == '__main__':
    tf.app.run()

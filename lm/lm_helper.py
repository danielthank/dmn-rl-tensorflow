
"""Utilities for parsing PTB & bAbI text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import json
import numpy as np

import tensorflow as tf

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().replace("\n", "<eos>").split()

def parse_task(lines):
    story = []
    for line in lines:
        line = line.strip()
        _, line = line.split(' ', 1)
        if '\t' in line:
            q, _, _ = line.split('\t')
            story.append(q)
        else:
            story.append(line)
    return story
def split_token(lines):
    ret = []
    for line in lines:
        for token in line.split(' '):
            token = token.strip()
            if not len(token):
                continue
            if token[-1] == '.' or token[-1] == '?':
                token = token[:-1]
            if len(token):
                ret.append(token.lower())
        ret.append("<eos>")
    return ret

def read_babi():
    tasks = list(range(1,21))
    all_train = []
    all_test = []
    for i in tasks:
        f_train = open('./babi/train/task_{}.txt'.format(i), 'r')
        train = parse_task(f_train.readlines())
        all_train += split_token(train)
        
        f_test = open('./babi/test/task_{}.txt'.format(i), 'r')
        test = parse_task(f_test.readlines())
        all_test += split_token(test)
    return all_train, all_test
        

def _build_vocab(filename):
  data = _read_words(filename)
  babi_train, babi_test = read_babi()
  counter = collections.Counter(data)
  counter.update(babi_train)
  counter.update(babi_test)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  data = ['where', 'is', 'mary', '<eos>', 'where', 'where', 'where', '<eos>']
  return [word_to_id[word] for word in data if word in word_to_id]

def babi_tokenizer(data, word_to_id):
    return [word_to_id[word] for word in data]

def raw_data_producer(data_path=None, save_dir=None):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  babi_train, babi_test = read_babi()
  print(len(train_data))
  train_data += babi_tokenizer(babi_train, word_to_id)
  print(len(train_data))
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  test_data += babi_tokenizer(babi_test, word_to_id)
  vocabulary = len(word_to_id)
  print('vocab_size: %d' % (vocabulary))
  if save_dir:
    filename = os.path.join(save_dir, 'vocab.json')
    print('[lm] Save vocabulary to %s' % (filename))
    with open(filename, 'w') as file:
      json.dump(word_to_id, file, indent=4)
  return train_data, valid_data, test_data, word_to_id
 
class LMDataSet:
    def __init__(self, batch_size, num_steps, data):
        self.data_len = len(data)
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.batch_len = self.data_len // self.batch_size # how many words in each batch
        self.epoch_size = (self.batch_len - 1) // self.num_steps # how many epoch to run full batch, avoid last word
        assert self.epoch_size > 0, 'not enough to form a epoch with {} data!'.format(self.data_len)
        self.data = np.array(data, dtype=np.int32)
        self.data = np.reshape(self.data[:self.batch_len * self.batch_size], [self.batch_size, self.batch_len])
        
    def get_batch(self):
        for i in range(self.epoch_size):
            x = self.data[:, i * self.num_steps: (i+1) * self.num_steps]
            y = self.data[:, i * self.num_steps + 1: (i+1) * self.num_steps + 1]
            yield (x, y)

def train_lm(model, params):
    train_data, valid_data, test_data, word_to_id = raw_data_producer(params.lm_ptb_path, params.save_dir)
    vocab_size = len(word_to_id)
    train_data = LMDataSet(params.lm_batch_size, params.lm_num_steps, train_data)
    valid_data = LMDataSet(params.lm_batch_size, params.lm_num_steps, valid_data)
    LM = model(vocab_size, params)
    LM.pre_train(train_data, valid_data)
    LM.save_params()

def test_lm(model, params):
    assert params.load_dir is not None, 'Please provide loading dir for LM'
    filename = os.path.join(params.load_dir, 'vocab.json')
    with open(filename, 'r') as file:
        word_to_id = json.load(file)
        vocab_size = len(word_to_id)
        print('[lm] Load %s with vocab size: %d' % (filename, vocab_size))
    LM = model(vocab_size, params)
    test_path = os.path.join(params.lm_ptb_path, "ptb.test.txt")
    babi_train, babi_test = read_babi()
    test_data = _file_to_word_ids(test_path, word_to_id)
    #test_data += babi_tokenizer(babi_test, word_to_id)
    test_data = LMDataSet(params.lm_batch_size, params.lm_num_steps, test_data)
    LM.eval(test_data, params.action)



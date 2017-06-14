# Common data loading utilities.
import pickle
import copy
import os
import numpy as np
import math
import random
class DataSet:
    def __init__(self, batch_size, xs, qs, ys, shuffle=True, name="dataset"):
        assert batch_size <= len(xs), "batch size cannot be greater than data size."
        self.name = name
        self.xs = np.array(xs, dtype=np.int32)
        self.qs = np.array(qs, dtype=np.int32)
        self.ys = np.array(ys, dtype=np.int32)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.count = len(self.xs)
        self.setup()

    def setup(self):
        self.indexes = list(range(self.count))  # used in shuffling
        self.current_index = 0
        self.num_batches = int(self.count / self.batch_size)
        self.reset()

    def next_batch(self,full_batch = True):
        '''
        if full batch is True, 
            return data with batch size
        else, 
            when the rest data is not enough, still see it as a batch
        '''
        # assert self.has_next_batch(full_batch), "End of epoch. Call 'complete_epoch()' to reset."
        if full_batch :
            from_, to = self.current_index, self.current_index + self.batch_size
        else:
            from_, to = self.current_index, min(self.current_index + self.batch_size,self.count)
        cur_idxs = self.indexes[from_:to]
        xs = self.xs[cur_idxs]
        qs = self.qs[cur_idxs]
        ys = self.ys[cur_idxs]
        self.current_index = to
        return xs, qs, ys

    def get_batch_cnt(self, cnt):
        if not self.has_next_batch(cnt):
            self.reset()
        from_, to = self.current_index, self.current_index + cnt
        cur_idxs = self.indexes[from_:to]
        xs = self.xs[cur_idxs]
        qs = self.qs[cur_idxs]
        ys = self.ys[cur_idxs]
        self.current_index += cnt
        return xs, qs, ys

    def get_bad_batch_cnt(self, cnt, vocab_size):
        if not self.has_next_batch(cnt):
            self.reset()
        from_, to = self.current_index, self.current_index + cnt
        cur_idxs = self.indexes[from_:to]
        xs = self.xs[cur_idxs]
        qs = self.qs[cur_idxs]
        ys = np.ones((cnt,), dtype=np.int32) # <go>
        choose = np.random.rand(*qs.shape) < 0.5
        choose[:, -1] = False # do not change <eos>
        qs = qs * (~choose) + np.random.randint(2, vocab_size, size=qs.shape) * choose # do not change to <eos> or <go>
        self.current_index += cnt
        return xs, qs, ys
    def get_random_cnt(self,cnt):
        choices = np.random.choice(self.count,cnt)
        xs = self.xs[choices]
        qs = self.qs[choices]
        ys = self.ys[choices]
        return xs, qs, ys

    """
    def has_next_batch(self, full_batch):
        if full_batch:
            return self.current_index + self.batch_size <= self.count
        else:
            return self.current_index < self.count
    """
    def has_next_batch(self, cnt):
        return self.current_index + cnt <= self.count

    def split_dataset(self, split_ratio):
        """ Splits a data set by split_ratio.
        (ex: split_ratio = 0.3 -> this set (70%) and splitted (30%))
        :param split_ratio: ratio of train data
        :return: val_set
        """
        end_index = int(self.count * (1 - split_ratio))

        # do not (deep) copy data - just modify index list!
        val_set = copy.copy(self)
        val_set.count = self.count - end_index
        val_set.indexes = list(range(end_index, self.count))
        val_set.num_batches = int(val_set.count / val_set.batch_size)
        self.count = end_index
        self.setup()
        return val_set
    
    def get_batch_num(self,full_batch = True):
        '''
            use this function is better than access num_batches directly
        '''
        if full_batch or self.count%self.batch_size == 0:
            return self.num_batches
        else:
            return self.num_batches + 1
    
    def reset(self):
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self,key):
        
        # do not (deep) copy data - just modify index list!
        val_set = copy.copy(self)
        if isinstance(key,slice):
            start = 0 if key.start == None else key.start
            stop = self.count if key.stop == None else key.stop
            step = 1 if key.step == None else key.step 
            
            val_set.count = int((stop - start)/step)
            val_set.indexes = [self.indexes[i] for i in range(start,stop,step)]
            val_set.num_batches = int(val_set.count / val_set.batch_size)
            return val_set
        else:
            raise NotImplementedError

       
class WordTable:
    def __init__(self, word2vec=None, embed_size=0):
        self.word2vec = word2vec
        self.word2idx = {'<eos>': 0, '<go>': 1}
        self.word2dc = {}
        self.idx2dc = {}
        self.all_doc_count = 0.
        self.idx2word = ['<eos>', '<go>']  # zero padding will be <eos>
        self.embed_size = embed_size

    def add_vocab(self, *words):
        """ Add vocabularies to dictionary. """
        for word in words:
            if self.word2vec and word not in self.word2vec:
                self._create_vector(word)

            if word not in self.word2idx:
                index = len(self.idx2word)
                self.word2idx[word] = index
                self.idx2word.append(word)

    def count_doc(self, *words):
        seen_words = []
        for word in words:
            if word not in self.word2dc:
                self.word2dc[word] = 1.
                self.idx2dc[self.word2idx[word]] = 1.
            elif word not in seen_words:
                self.word2dc[word] += 1.
                self.idx2dc[self.word2idx[word]] += 1.
                seen_words.append(word)
        self.all_doc_count += 1.

    def find_keyterm_by_word(self, *words):
        doc_counts = np.array([self.word2dc.get(word, np.inf) for word in words])
        keyterm = words[np.argmin(doc_counts)]
        return keyterm

    def find_keyterm_by_idx(self, *words):
        doc_counts = np.array([self.idx2dc.get(word, np.inf) for word in words])
        keyterm = words[np.argmin(doc_counts)]
        return keyterm

    def vectorize(self, word):
        """ Converts word to vector.
        :param word: string
        :return: 1-D array (vector)
        """
        self.add_vocab(word)
        return self.word2vec[word]

    def _create_vector(self, word):
        # if the word is missing from Glove, create some fake vector and store in glove!
        vector = np.random.uniform(0.0, 1.0, (self.embed_size,))
        self.word2vec[word] = vector
        print("create_vector => %s is missing" % word)
        return vector
    """
    def word_to_index(self, word):
        self.add_vocab(word)
        return self.word2idx[word]

    def index_to_word(self, index):
        return self.idx2word[index]
    """

    @property
    def vocab_size(self):
        return len(self.idx2word)


def load_glove(dim):
    """ Loads GloVe data.
    :param dim: word vector size (50, 100, 200)
    :return: GloVe word table
    """
    word2vec = {}

    path = "data/glove/glove.6B." + str(dim) + 'd'
    if os.path.exists(path + '.cache'):
        with open(path + '.cache', 'rb') as cache_file:
            word2vec = pickle.load(cache_file)

    else:
        # Load n create cache
        with open(path + '.txt') as f:
            for line in f:
                l = line.split()
                word2vec[l[0]] = [float(x) for x in l[1:]]

        with open(path + '.cache', 'wb') as cache_file:
            pickle.dump(word2vec, cache_file)

    print("Loaded Glove data")
    return word2vec

from collections import deque
import warnings
import numpy as np

class QuestionMemory():
    def __init__(self, element_shape, max_len, dtype=np.float32):
        self.buffer = np.zeros((max_len,)+element_shape, dtype=dtype)
        self.max_len = max_len
        self.idx = 0
        self.full = False

    def append(self, questions):
        questions = np.array(questions)
        num = questions.shape[0]
        if num == 0:
            warnings.warn("Appending an empty array!", UserWarning)
            return
        if not self.full and self.idx+num >= self.max_len:
            self.full = True
        if num > self.max_len:
            choice = np.random.choice(num, self.max_len, replace=False)
            self.buffer = questions[choice]
            self.idx = 0
        elif num == self.max_len:
            self.buffer = questions
            self.idx = 0
        else:
            add_idx = (self.idx + np.arange(num)) % self.max_len
            self.buffer[add_idx] = questions
            self.idx = add_idx[-1] + 1

    def all(self):
        if self.full:
            return self.buffer
        else:
            return self.buffer[:self.idx]

    def reset(self):
        self.buffer = np.zeros((self.max_len,)+self.buffer.shape[1:], dtype=self.buffer.dtype)
        self.idx = 0
        self.full = False

    def __len__(self):
        if self.full:
            return self.max_len
        else:
            return self.idx
    
    def __getitem__(self, key):
        if self.full:
            return self.buffer[key]
        else:
            return self.buffer[:self.idx][key]

    def __setitem__(self, key, value):
        if self.full:
            self.buffer[key] = value
        else:
            self.buffer[:self.idx][key] = value


def test():
    max_len = 100000
    qm = QuestionMemory((1,), max_len)
    for i in range(40):
        qm.append(np.zeros((10000,1), dtype='float32'))
        qm.all()


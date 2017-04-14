from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

def get_sequence_length(sequence, scope=None):
    """
    This is a hacky way of determining the actual length of a sequence that has been padded with zeros.
    """
    with tf.variable_scope(scope, 'SequenceLength'):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), axis=[-1]))
        length = tf.cast(tf.reduce_sum(used, axis=[-1]), tf.int32)
        return length

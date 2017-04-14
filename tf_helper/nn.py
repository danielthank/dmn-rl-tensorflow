import math
import tensorflow as tf
import numpy as np

def weight(name, shape, init='he', range=None):
    """ Initializes weight.
    :param name: Variable name
    :param shape: Tensor shape
    :param init: Init mode. xavier / normal / uniform / he (default is 'he')
    :param range:
    :return: Variable
    """
    initializer = tf.constant_initializer()
    if init == 'xavier':
        fan_in, fan_out = _get_dims(shape)
        range = math.sqrt(6.0 / (fan_in + fan_out))
        initializer = tf.random_uniform_initializer(-range, range)

    elif init == 'he':
        fan_in, _ = _get_dims(shape)
        std = math.sqrt(2.0 / fan_in)
        initializer = tf.random_normal_initializer(stddev=std)

    elif init == 'normal':
        initializer = tf.random_normal_initializer(stddev=0.1)

    elif init == 'uniform':
        if range is None:
            raise ValueError("range must not be None if uniform init is used.")
        initializer = tf.random_uniform_initializer(-range, range)

    var = tf.get_variable(name, shape, initializer=initializer)
    tf.add_to_collection('l2', tf.nn.l2_loss(var))  # Add L2 Loss
    return var

def variable_summary(vars):
    #collections = ["PRE_SUMM", "QA_SUMM", "RL_SUMM"]
    collections = ["VAR_SUMM"]
    for var in vars:
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean/' + var.name, mean, collections=collections)
            tf.summary.scalar('stddev/' + var.name, tf.sqrt(tf.reduce_mean(tf.square(var - mean))), collections=collections)
            tf.summary.histogram(var.name, var, collections=collections)

def _get_dims(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[:-1])
    fan_out = shape[1] if len(shape) == 2 else shape[-1]
    return fan_in, fan_out


def bias(name, dim, initial_value=0.0):
    """ Initializes bias parameter.
    :param name: Variable name
    :param dim: Tensor size (list or int)
    :param initial_value: Initial bias term
    :return: Variable
    """
    dims = dim if isinstance(dim, list) else [dim]
    var = tf.get_variable(name, dims, initializer=tf.constant_initializer(initial_value))
    return var

"""
def batch_norm(x, is_training):
    #Batch normalization.
    #:param x: Tensor
    #:param is_training: boolean tf.Variable, true indicates training phase
    #:return: batch-normalized tensor
    #
    with tf.variable_scope('BatchNorm'):
        # calculate dimensions (from tf.contrib.layers.batch_norm)
        inputs_shape = x.get_shape()
        axis = list(range(len(inputs_shape) - 1))
        param_shape = inputs_shape[-1:]

        beta = tf.get_variable('beta', param_shape, initializer=tf.constant_initializer(0.))
        gamma = tf.get_variable('gamma', param_shape, initializer=tf.constant_initializer(1.))
        batch_mean, batch_var = tf.nn.moments(x, axis)
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed
"""

def batch_norm(x, is_training):
    outputs = tf.contrib.layers.batch_norm(x,
                                           decay=0.9,
                                           is_training=is_training,
                                           center=True,
                                           scale=True,
                                           activation_fn=tf.nn.relu,
                                           updates_collections=None,
                                           scope='BatchNorm')
    return outputs

def dropout(x, keep_prob, is_training):
    """ Apply dropout.
    :param x: Tensor
    :param keep_prob: float, Dropout rate.
    :param is_training: boolean tf.Varialbe, true indicates training phase
    :return: dropout applied tensor
    """
    return tf.cond(is_training, lambda: tf.nn.dropout(x, keep_prob), lambda: x)

def conv(x, filter, is_training):
    l = tf.nn.conv2d(x, filter, strides=[1, 1, 1, 1], padding='SAME')
    l = batch_norm(l, is_training)
    return tf.nn.relu(l)

def flatten(x):
    return tf.reshape(x, [-1])

def fully_connected(input, num_neurons, name, is_training):
    input_size = input.get_shape()[1]
    w = weight(name, [input_size, num_neurons], init='he')
    l = tf.matmul(input, w)
    l = batch_norm(l, is_training)
    return tf.nn.relu(l)

def sample_gumbel(shape, eps=1e-20): 
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape,minval=0,maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature): 
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, axis, temperature=1.0):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar, default to 1.0
    Returns:
        [batch_size] index sample from the Gumbel-Softmax distribution.
    """
    y = gumbel_softmax_sample(logits, temperature)
    max_index = tf.argmax(y, axis=axis)
    return max_index

def prelu(features, initializer=None, scope=None):
    """
    Implementation of [Parametric ReLU](https://arxiv.org/abs/1502.01852) borrowed from Keras.
    """
    with tf.variable_scope(scope, 'PReLU', initializer=initializer):
        alpha = tf.get_variable('alpha', features.get_shape().as_list()[1:])
        pos = tf.nn.relu(features)
        neg = alpha * (features - tf.abs(features)) * 0.5
        return pos + neg

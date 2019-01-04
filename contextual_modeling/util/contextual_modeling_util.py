import numpy as np
import tensorflow as tf

from util.default_util import *

__all__ = ["create_variable_initializer", "create_weight_regularizer", "create_activation_function",
           "softmax_with_mask", "generate_masked_data"]

def create_variable_initializer(initializer_type,
                                random_seed=None,
                                data_type=tf.float32):
    """create variable initializer"""
    if initializer_type == "zero":
        initializer = tf.zeros_initializer
    elif initializer_type == "orthogonal":
        initializer = tf.orthogonal_initializer(seed=random_seed, dtype=data_type)
    elif initializer_type == "random_uniform":
        initializer = tf.random_uniform_initializer(seed=random_seed, dtype=data_type)
    elif initializer_type == "glorot_uniform":
        initializer = tf.glorot_uniform_initializer(seed=random_seed, dtype=data_type)
    elif initializer_type == "xavier_uniform":
        initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=random_seed, dtype=tf.float32)
    elif initializer_type == "random_normal":
        initializer = tf.random_normal_initializer(seed=random_seed, dtype=data_type)
    elif initializer_type == "truncated_normal":
        initializer = tf.truncated_normal_initializer(seed=random_seed, dtype=data_type)
    elif initializer_type == "glorot_normal":
        initializer = tf.glorot_normal_initializer(seed=random_seed, dtype=data_type)
    elif initializer_type == "xavier_normal":
        initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=random_seed, dtype=tf.float32)
    else:
        initializer = None
    
    return initializer

def create_weight_regularizer(regularizer_type,
                              scale):
    """create weight regularizer"""
    if regularizer_type == "l1":
        regularizer = tf.contrib.layers.l1_regularizer(scale)
    elif regularizer_type == "l2":
        regularizer = tf.contrib.layers.l2_regularizer(scale)
    else:
        regularizer = None
    
    return regularizer

def create_activation_function(activation):
    """create activation function"""
    if activation == "relu":
        activation_function = tf.nn.relu
    elif activation == "relu6":
        activation_function = tf.nn.relu6
    elif activation == "leaky_relu":
        activation_function = tf.nn.leaky_relu
    elif activation == "elu":
        activation_function = tf.nn.elu
    elif activation == "crelu":
        activation_function = tf.nn.crelu
    elif activation == "selu":
        activation_function = tf.nn.selu
    elif activation == "gelu":
        activation_function = gelu
    elif activation == "tanh":
        activation_function = tf.nn.tanh
    elif activation == "sigmoid":
        activation_function = tf.nn.sigmoid
    elif activation == "softplus":
        activation_function = tf.nn.softplus
    else:
        activation_function = None
    
    return activation_function

def softmax_with_mask(input_data,
                      input_mask,
                      axis=-1):
    """compute softmax with masking"""    
    return tf.nn.softmax(input_data * input_mask + MIN_FLOAT * (1 - input_mask), axis=axis)

def generate_masked_data(input_data,
                         input_mask):
    """generate masked data"""
    return input_data * input_mask + MIN_FLOAT * (1 - input_mask)

def gelu(input_tensor):
    """Gaussian Error Linear Unit"""
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf

import numpy as np
import tensorflow as tf

from util.default_util import *
from util.contextual_modeling_util import *

__all__ = ["SinusoidPosition", "AbsolutePosition"]

class SinusoidPosition(object):
    """sinusoid position layer"""
    def __init__(self,
                 min_time_scale,
                 max_time_scale,
                 num_gpus=1,
                 default_gpu_id=0,
                 scope="sin_pos"):
        """initialize sinusoid position layer"""
        self.min_time_scale = min_time_scale
        self.max_time_scale = max_time_scale
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call sinusoid position layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_shape = tf.shape(input_data)
            length = input_shape[-2]
            channel = input_shape[-1]
            num_time_scale = channel // 2
            position = tf.to_float(tf.range(length))
            log_time_scale = tf.log(float(self.max_time_scale) / float(self.min_time_scale)) / (tf.to_float(num_time_scale) - 1)
            inv_time_scale = float(self.min_time_scale) * tf.exp(-1.0 * log_time_scale * tf.to_float(tf.range(num_time_scale)))
            scaled_time = tf.expand_dims(position, axis=1) * tf.expand_dims(inv_time_scale, axis=0)
            signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
            signal = tf.pad(signal, paddings=[[0, 0], [0, tf.mod(channel, 2)]])
            signal = tf.reshape(signal, shape=[1, length, channel])
            
            output_signal = input_data + signal
            output_mask = input_mask
        
        return output_signal, output_mask

class AbsolutePosition(object):
    """absolute position layer"""
    def __init__(self,
                 unit_dim,
                 max_length,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="abs_pos"):
        """initialize absolute position layer"""
        self.unit_dim = unit_dim
        self.max_length = max_length
        self.random_seed = random_seed
        self.regularizer = regularizer
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            weight_initializer = create_variable_initializer("glorot_uniform", self.random_seed)
            self.position_embedding = tf.get_variable("position_embedding", shape=[1, self.max_length, self.unit_dim],
                initializer=weight_initializer, regularizer=self.regularizer, trainable=self.trainable, dtype=tf.float32)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call absolute position layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_shape = tf.shape(input_data)
            max_length = input_shape[-2]
            position_embedding = self.position_embedding[:,:max_length,:]
            output_signal = input_data + position_embedding
            output_mask = input_mask
        
        return output_signal, output_mask

import numpy as np
import tensorflow as tf

from util.default_util import *
from util.contextual_modeling_util import *

from layer.basic import *

__all__ = ["Conv1D", "Conv3D", "SeparableConv1D", "MultiConv", "StackedConv", "StackedMultiConv"]

class Conv1D(object):
    """1d convolution layer"""
    def __init__(self,
                 num_channel,
                 num_filter,
                 window_size,
                 stride_size,
                 padding_type,
                 activation,
                 dropout,
                 layer_dropout=0.0,
                 layer_norm=False,
                 residual_connect=False,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="conv_1d"):
        """initialize 1d convolution layer"""
        self.num_channel = num_channel
        self.num_filter = num_filter
        self.window_size = window_size
        self.stride_size = stride_size
        self.padding_type = padding_type
        self.activation = activation
        self.dropout = dropout
        self.layer_dropout = layer_dropout
        self.layer_norm = layer_norm
        self.residual_connect = residual_connect
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            weight_initializer = (create_variable_initializer("variance_scaling", self.random_seed)
                if self.activation == "relu" else create_variable_initializer("glorot_uniform", self.random_seed))
            bias_initializer = create_variable_initializer("zero")
            conv_activation = create_activation_function(self.activation)
            self.conv_layer = tf.layers.Conv1D(filters=self.num_filter, kernel_size=window_size,
                strides=stride_size, padding=self.padding_type, activation=conv_activation, use_bias=True,
                kernel_initializer=weight_initializer, bias_initializer=bias_initializer,
                kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer, trainable=trainable)
            
            self.dropout_layer = Dropout(rate=self.dropout, num_gpus=num_gpus,
                default_gpu_id=default_gpu_id, random_seed=self.random_seed)
            
            if self.layer_norm == True:
                self.norm_layer = LayerNorm(layer_dim=self.num_channel, num_gpus=num_gpus,
                    default_gpu_id=default_gpu_id, regularizer=self.regularizer, trainable=self.trainable)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call 1d convolution layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_data_shape = tf.shape(input_data)
            input_mask_shape = tf.shape(input_mask)
            shape_size = len(input_data.get_shape().as_list())
            if shape_size > 3:
                input_conv = tf.reshape(input_data, shape=tf.concat([[-1], input_data_shape[-2:]], axis=0))
                input_conv_mask = tf.reshape(input_mask, shape=tf.concat([[-1], input_mask_shape[-2:]], axis=0))
            else:
                input_conv = input_data
                input_conv_mask = input_mask
            
            input_conv, input_conv_mask = self.dropout_layer(input_conv, input_conv_mask)
            
            if self.layer_norm == True:
                input_conv, input_conv_mask = self.norm_layer(input_conv, input_conv_mask)
            
            input_conv = self.conv_layer(input_conv)
            
            if self.residual_connect == True:
                output_conv, output_mask = tf.cond(tf.random_uniform([]) < self.layer_dropout,
                    lambda: (input_data, input_mask),
                    lambda: (input_conv + input_data, input_conv_mask * input_mask))
            else:
                output_conv = input_conv
                output_mask = input_mask
            
            if shape_size > 3:
                output_conv_shape = tf.shape(output_conv)
                output_mask_shape = tf.shape(output_mask)
                output_conv = tf.reshape(output_conv,
                    shape=tf.concat([input_data_shape[:-2], output_conv_shape[-2:]], axis=0))
                output_mask = tf.reshape(output_mask,
                    shape=tf.concat([input_mask_shape[:-2], output_mask_shape[-2:]], axis=0))
        
        return output_conv, output_mask

class Conv3D(object):
    """3d convolution layer"""
    def __init__(self,
                 num_channel,
                 num_filter,
                 window_size,
                 stride_size,
                 padding_type,
                 activation,
                 dropout,
                 layer_dropout=0.0,
                 layer_norm=False,
                 residual_connect=False,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="conv_3d"):
        """initialize 3d convolution layer"""
        self.num_channel = num_channel
        self.num_filter = num_filter
        self.window_size = window_size
        self.stride_size = stride_size
        self.padding_type = padding_type
        self.activation = activation
        self.dropout = dropout
        self.layer_dropout = layer_dropout
        self.layer_norm = layer_norm
        self.residual_connect = residual_connect
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            weight_initializer = (create_variable_initializer("variance_scaling", self.random_seed)
                if self.activation == "relu" else create_variable_initializer("glorot_uniform", self.random_seed))
            bias_initializer = create_variable_initializer("zero")
            conv_activation = create_activation_function(self.activation)
            self.conv_layer = tf.layers.Conv3D(filters=self.num_filter, kernel_size=window_size,
                strides=stride_size, padding=self.padding_type, activation=conv_activation, use_bias=True,
                kernel_initializer=weight_initializer, bias_initializer=bias_initializer,
                kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer, trainable=trainable)
            
            self.dropout_layer = Dropout(rate=self.dropout, num_gpus=num_gpus,
                default_gpu_id=default_gpu_id, random_seed=self.random_seed)
            
            if self.layer_norm == True:
                self.norm_layer = LayerNorm(layer_dim=self.num_channel, num_gpus=num_gpus,
                    default_gpu_id=default_gpu_id, regularizer=self.regularizer, trainable=self.trainable)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call 3d convolution layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_data_shape = tf.shape(input_data)
            input_mask_shape = tf.shape(input_mask)
            shape_size = len(input_data.get_shape().as_list())
            if shape_size > 5:
                input_conv = tf.reshape(input_data, shape=tf.concat([[-1], input_data_shape[-4:]], axis=0))
                input_conv_mask = tf.reshape(input_mask, shape=tf.concat([[-1], input_mask_shape[-4:]], axis=0))
            else:
                input_conv = input_data
                input_conv_mask = input_mask
            
            input_conv, input_conv_mask = self.dropout_layer(input_conv, input_conv_mask)
            
            if self.layer_norm == True:
                input_conv, input_conv_mask = self.norm_layer(input_conv, input_conv_mask)
            
            input_conv = self.conv_layer(input_conv)
            
            if self.residual_connect == True:
                output_conv, output_mask = tf.cond(tf.random_uniform([]) < self.layer_dropout,
                    lambda: (input_data, input_mask),
                    lambda: (input_conv + input_data, input_conv_mask * input_mask))
            else:
                output_conv = input_conv
                output_mask = input_mask
            
            if shape_size > 5:
                output_conv_shape = tf.shape(output_conv)
                output_mask_shape = tf.shape(output_mask)
                output_conv = tf.reshape(output_conv,
                    shape=tf.concat([input_data_shape[:-4], output_conv_shape[-4:]], axis=0))
                output_mask = tf.reshape(output_mask,
                    shape=tf.concat([input_mask_shape[:-4], output_mask_shape[-4:]], axis=0))
        
        return output_conv, output_mask

class SeparableConv1D(object):
    """depthwise-separable 1d convolution layer"""
    def __init__(self,
                 num_channel,
                 num_filter,
                 window_size,
                 stride_size,
                 padding_type,
                 activation,
                 dropout,
                 layer_dropout=0.0,
                 layer_norm=False,
                 residual_connect=False,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="sep_conv_1d"):
        """initialize depthwise-separable 1d convolution layer"""
        self.num_channel = num_channel
        self.num_filter = num_filter
        self.window_size = window_size
        self.stride_size = stride_size
        self.padding_type = padding_type
        self.activation = activation
        self.dropout = dropout
        self.layer_dropout = layer_dropout
        self.layer_norm = layer_norm
        self.residual_connect = residual_connect
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            weight_initializer = (create_variable_initializer("variance_scaling", self.random_seed)
                if self.activation == "relu" else create_variable_initializer("glorot_uniform", self.random_seed))
            bias_initializer = create_variable_initializer("zero")
            self.depthwise_filter = tf.get_variable("depthwise_filter",
                shape=[1, self.window_size, self.num_channel, 1], initializer=weight_initializer,
                regularizer=self.regularizer, trainable=self.trainable, dtype=tf.float32)
            self.pointwise_filter = tf.get_variable("pointwise_filter",
                shape=[1, 1, self.num_channel * 1, self.num_filter], initializer=weight_initializer,
                regularizer=self.regularizer, trainable=self.trainable, dtype=tf.float32)
            self.separable_bias = tf.get_variable("separable_bias", shape=[self.num_filter], initializer=bias_initializer,
                regularizer=self.regularizer, trainable=trainable, dtype=tf.float32)
            
            self.strides = [1, 1, self.stride_size, 1]
            self.conv_activation = create_activation_function(self.activation)
            
            self.dropout_layer = Dropout(rate=self.dropout, num_gpus=num_gpus,
                default_gpu_id=default_gpu_id, random_seed=self.random_seed)
            
            if self.layer_norm == True:
                self.norm_layer = LayerNorm(layer_dim=self.num_channel, num_gpus=num_gpus,
                    default_gpu_id=default_gpu_id, regularizer=self.regularizer, trainable=self.trainable)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call depthwise-separable 1d convolution layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_data_shape = tf.shape(input_data)
            input_mask_shape = tf.shape(input_mask)
            shape_size = len(input_data.get_shape().as_list())
            if shape_size > 3:
                input_conv = tf.reshape(input_data, shape=tf.concat([[-1], input_data_shape[-2:]], axis=0))
                input_conv_mask = tf.reshape(input_mask, shape=tf.concat([[-1], input_mask_shape[-2:]], axis=0))
            else:
                input_conv = input_data
                input_conv_mask = input_mask
            
            input_conv, input_conv_mask = self.dropout_layer(input_conv, input_conv_mask)
            
            if self.layer_norm == True:
                input_conv, input_conv_mask = self.norm_layer(input_conv, input_conv_mask)
            
            input_conv = tf.expand_dims(input_conv, axis=1)
            input_conv = tf.nn.separable_conv2d(input_conv, self.depthwise_filter,
                self.pointwise_filter, self.strides, self.padding_type)
            input_conv = tf.squeeze(input_conv, axis=1)
            
            input_conv = input_conv + self.separable_bias
            if self.conv_activation != None:
                input_conv = self.conv_activation(input_conv)
            
            if self.residual_connect == True:
                output_conv, output_mask = tf.cond(tf.random_uniform([]) < self.layer_dropout,
                    lambda: (input_data, input_mask),
                    lambda: (input_conv + input_data, input_conv_mask * input_mask))
            else:
                output_conv = input_conv
                output_mask = input_mask
            
            if shape_size > 3:
                output_conv_shape = tf.shape(output_conv)
                output_mask_shape = tf.shape(output_mask)
                output_conv = tf.reshape(output_conv,
                    shape=tf.concat([input_data_shape[:-2], output_conv_shape[-2:]], axis=0))
                output_mask = tf.reshape(output_mask,
                    shape=tf.concat([input_mask_shape[:-2], output_mask_shape[-2:]], axis=0))
        
        return output_conv, output_mask

class MultiConv(object):
    """multi-window convolution layer"""
    def __init__(self,
                 layer_creator,
                 num_channel,
                 num_filter,
                 window_size,
                 stride_size,
                 padding_type,
                 activation,
                 dropout,
                 layer_dropout=0.0,
                 layer_norm=False,
                 residual_connect=False,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="multi_conv"):
        """initialize multi-window convolution layer"""
        self.layer_creator = layer_creator
        self.num_channel = num_channel
        self.num_filter = num_filter
        self.window_size = window_size
        self.stride_size = stride_size
        self.padding_type = padding_type
        self.activation = activation
        self.dropout = dropout
        self.layer_dropout = layer_dropout
        self.layer_norm = layer_norm
        self.residual_connect = residual_connect
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            self.conv_layer_list = []
            for i in range(len(self.window_size)):
                layer_scope = "window_{0}".format(i)
                layer_default_gpu_id = self.default_gpu_id
                conv_layer = self.layer_creator(num_channel=self.num_channel, num_filter=self.num_filter,
                    window_size=self.window_size[i], stride_size=self.stride_size, padding_type=self.padding_type,
                    activation=self.activation, dropout=self.dropout, layer_dropout=self.layer_dropout,
                    layer_norm=self.layer_norm, residual_connect=self.residual_connect, num_gpus=self.num_gpus,
                    default_gpu_id=layer_default_gpu_id, regularizer=self.regularizer, random_seed=self.random_seed,
                    trainable=self.trainable, scope=layer_scope)
                self.conv_layer_list.append(conv_layer)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call multi-window depthwise-separable convolution layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_conv_list = []
            input_conv_mask_list = []
            for conv_layer in self.conv_layer_list:
                input_conv, input_conv_mask = conv_layer(input_data, input_mask)
                input_conv_list.append(input_conv)
                input_conv_mask_list.append(input_conv_mask)
            
            output_conv = tf.concat(input_conv_list, axis=-1)
            output_mask = tf.reduce_max(tf.concat(input_conv_mask_list, axis=-1), axis=-1, keepdims=True)
        
        return output_conv, output_mask

class StackedConv(object):
    """stacked convolution layer"""
    def __init__(self,
                 layer_creator,
                 num_layer,
                 num_channel,
                 num_filter,
                 window_size,
                 stride_size,
                 padding_type,
                 activation,
                 dropout,
                 layer_dropout=None,
                 layer_norm=False,
                 residual_connect=False,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="stacked_conv"):
        """initialize stacked convolution layer"""
        self.layer_creator = layer_creator
        self.num_layer = num_layer
        self.num_channel = num_channel
        self.num_filter = num_filter
        self.window_size = window_size
        self.stride_size = stride_size
        self.padding_type = padding_type
        self.activation = activation
        self.dropout = dropout
        self.layer_dropout = layer_dropout
        self.layer_norm = layer_norm
        self.residual_connect = residual_connect
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            self.conv_layer_list = []
            for i in range(self.num_layer):
                layer_scope = "layer_{0}".format(i)
                layer_default_gpu_id = self.default_gpu_id
                sublayer_dropout = self.dropout[i] if self.dropout != None else 0.0
                sublayer_layer_dropout = self.layer_dropout[i] if self.layer_dropout != None else 0.0
                conv_layer = self.layer_creator(num_channel=self.num_channel, num_filter=self.num_filter,
                    window_size=self.window_size, stride_size=self.stride_size, padding_type=self.padding_type,
                    activation=self.activation, dropout=sublayer_dropout, layer_dropout=sublayer_layer_dropout,
                    layer_norm=self.layer_norm, residual_connect=self.residual_connect, num_gpus=self.num_gpus,
                    default_gpu_id=layer_default_gpu_id, regularizer=self.regularizer, random_seed=self.random_seed,
                    trainable=self.trainable, scope=layer_scope)
                self.conv_layer_list.append(conv_layer)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call stacked convolution layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_conv = input_data
            input_conv_mask = input_mask
            
            for conv_layer in self.conv_layer_list:
                input_conv, input_conv_mask = conv_layer(input_conv, input_conv_mask)
            
            output_conv = input_conv
            output_mask = input_conv_mask
        
        return output_conv, output_mask

class StackedMultiConv(object):
    """stacked multi-window convolution layer"""
    def __init__(self,
                 layer_creator,
                 num_layer,
                 num_channel,
                 num_filter,
                 window_size,
                 stride_size,
                 padding_type,
                 activation,
                 dropout,
                 layer_dropout=None,
                 layer_norm=False,
                 residual_connect=False,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="stacked_multi_conv"):
        """initialize stacked multi-window convolution layer"""
        self.layer_creator = layer_creator
        self.num_layer = num_layer
        self.num_channel = num_channel
        self.num_filter = num_filter
        self.window_size = window_size
        self.stride_size = stride_size
        self.padding_type = padding_type
        self.activation = activation
        self.dropout = dropout
        self.layer_dropout = layer_dropout
        self.layer_norm = layer_norm
        self.residual_connect = residual_connect
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        self.device_spec = get_device_spec(default_gpu_id, num_gpus)
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            self.conv_layer_list = []
            for i in range(self.num_layer):
                layer_scope = "layer_{0}".format(i)
                layer_default_gpu_id = self.default_gpu_id
                sublayer_dropout = self.dropout[i] if self.dropout != None else 0.0
                sublayer_layer_dropout = self.layer_dropout[i] if self.layer_dropout != None else 0.0
                conv_layer = MultiConv(layer_creator=self.layer_creator, num_channel=self.num_channel,
                    num_filter=self.num_filter, window_size=self.window_size, stride_size=self.stride_size, 
                    padding_type=self.padding_type, activation=self.activation, dropout=sublayer_dropout, 
                    layer_dropout=sublayer_layer_dropout, layer_norm=self.layer_norm, residual_connect=self.residual_connect,
                    num_gpus=self.num_gpus, default_gpu_id=layer_default_gpu_id, regularizer=self.regularizer,
                    random_seed=self.random_seed, trainable=self.trainable, scope=layer_scope)
                self.conv_layer_list.append(conv_layer)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call stacked multi-window convolution layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE), tf.device(self.device_spec):
            input_conv = input_data
            input_conv_mask = input_mask
            
            for conv_layer in self.conv_layer_list:
                input_conv, input_conv_mask = conv_layer(input_conv, input_conv_mask)
            
            output_conv = input_conv
            output_mask = input_conv_mask
        
        return output_conv, output_mask

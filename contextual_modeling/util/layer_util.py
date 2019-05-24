import numpy as np
import tensorflow as tf

from layer.basic import *
from layer.embedding import *
from layer.position import *
from layer.convolution import *
from layer.pooling import *
from layer.dense import *
from layer.highway import *
from layer.recurrent import *
from layer.attention import *

__all__ = ["AttentionMechanism", "create_dropout_layer", "create_embedding_layer", "create_position_layer", "create_convolution_layer",
           "create_pooling_layer", "create_dense_layer", "create_highway_layer", "create_recurrent_layer", "create_attention_layer"]

def create_dropout_layer(dropout,
                         num_gpus,
                         default_gpu_id,
                         random_seed):
    """create dropout layer"""
    dropout_layer = Dropout(rate=dropout, num_gpus=num_gpus, default_gpu_id=default_gpu_id, random_seed=random_seed)
    return dropout_layer

def create_embedding_layer(vocab_size,
                           embed_dim,
                           embed_data,
                           pretrained,
                           num_gpus,
                           default_gpu_id,
                           regularizer,
                           random_seed,
                           trainable):
    """create embedding layer"""
    if pretrained == True:
        embed_layer = PretrainedEmbedding(vocab_size=vocab_size, embed_dim=embed_dim, embed_data=embed_data,
            num_gpus=num_gpus, default_gpu_id=default_gpu_id, regularizer=regularizer, trainable=trainable)
    else:
        embed_layer = Embedding(vocab_size=vocab_size, embed_dim=embed_dim,
            num_gpus=num_gpus, default_gpu_id=default_gpu_id, regularizer=regularizer, random_seed=random_seed, trainable=trainable)
    
    return embed_layer

def create_position_layer(position_type,
                          unit_dim,
                          max_length,
                          min_time_scale,
                          max_time_scale,
                          num_gpus,
                          default_gpu_id,
                          regularizer,
                          random_seed,
                          trainable):
    """create position layer"""
    scope = "position/{0}".format(position_type)
    if position_type == "sin_pos":
        position_layer = SinusoidPosition(min_time_scale=min_time_scale, max_time_scale=max_time_scale,
            num_gpus=num_gpus, default_gpu_id=default_gpu_id, scope=scope)
    elif position_type == "abs_pos":
        position_layer = AbsolutePosition(unit_dim=unit_dim, max_length=max_length,
            num_gpus=num_gpus, default_gpu_id=default_gpu_id, regularizer=regularizer,
            random_seed=random_seed, trainable=trainable, scope=scope)
    else:
        raise ValueError("unsupported position type {0}".format(position_type))
    
    return position_layer

def create_convolution_layer(conv_type,
                             num_layer,
                             num_channel,
                             num_filter,
                             window_size,
                             stride_size,
                             padding_type,
                             activation,
                             dropout,
                             layer_dropout,
                             layer_norm,
                             residual_connect,
                             use_bias,
                             num_gpus,
                             default_gpu_id,
                             regularizer,
                             random_seed,
                             trainable):
    """create convolution layer"""
    scope = "conv/{0}".format(conv_type)
    if conv_type == "stacked_1d":
        conv_layer = StackedConv(layer_creator=Conv1D, num_layer=num_layer, num_channel=num_channel,
            num_filter=num_filter, window_size=window_size, stride_size=stride_size, padding_type=padding_type,
            activation=activation, dropout=dropout, layer_dropout=layer_dropout, layer_norm=layer_norm,
            residual_connect=residual_connect, use_bias=use_bias, num_gpus=num_gpus, default_gpu_id=default_gpu_id,
            regularizer=regularizer, random_seed=random_seed, trainable=trainable, scope=scope)
    elif conv_type == "stacked_3d":
        conv_layer = StackedConv(layer_creator=Conv3D, num_layer=num_layer, num_channel=num_channel,
            num_filter=num_filter, window_size=window_size, stride_size=stride_size, padding_type=padding_type,
            activation=activation, dropout=dropout, layer_dropout=layer_dropout, layer_norm=layer_norm,
            residual_connect=residual_connect, use_bias=use_bias, num_gpus=num_gpus, default_gpu_id=default_gpu_id,
            regularizer=regularizer, random_seed=random_seed, trainable=trainable, scope=scope)
    elif conv_type == "stacked_sep_1d":
        conv_layer = StackedConv(layer_creator=SeparableConv1D, num_layer=num_layer, num_channel=num_channel,
            num_filter=num_filter, window_size=window_size, stride_size=stride_size, padding_type=padding_type,
            activation=activation, dropout=dropout, layer_dropout=layer_dropout, layer_norm=layer_norm,
            residual_connect=residual_connect, use_bias=use_bias, num_gpus=num_gpus, default_gpu_id=default_gpu_id,
            regularizer=regularizer, random_seed=random_seed, trainable=trainable, scope=scope)
    elif conv_type == "stacked_multi_1d":
        conv_layer = StackedMultiConv(layer_creator=Conv1D, num_layer=num_layer, num_channel=num_channel,
            num_filter=num_filter, window_size=window_size, stride_size=stride_size, padding_type=padding_type,
            activation=activation, dropout=dropout, layer_dropout=layer_dropout, layer_norm=layer_norm,
            residual_connect=residual_connect, use_bias=use_bias, num_gpus=num_gpus, default_gpu_id=default_gpu_id,
            regularizer=regularizer, random_seed=random_seed, trainable=trainable, scope=scope)
    elif conv_type == "stacked_multi_sep_1d":
        conv_layer = StackedMultiConv(layer_creator=SeparableConv1D, num_layer=num_layer, num_channel=num_channel,
            num_filter=num_filter, window_size=window_size, stride_size=stride_size, padding_type=padding_type,
            activation=activation, dropout=dropout, layer_dropout=layer_dropout, layer_norm=layer_norm,
            residual_connect=residual_connect, use_bias=use_bias, num_gpus=num_gpus, default_gpu_id=default_gpu_id,
            regularizer=regularizer, random_seed=random_seed, trainable=trainable, scope=scope)
    else:
        raise ValueError("unsupported convolution type {0}".format(conv_type))
    
    return conv_layer

def create_pooling_layer(pooling_type,
                         window_size,
                         stride_size,
                         num_gpus,
                         default_gpu_id):
    """create pooling layer"""
    scope = "pooling/{0}".format(pooling_type)
    if pooling_type == "max":
        pooling_layer = MaxPooling(num_gpus=num_gpus, default_gpu_id=default_gpu_id, scope=scope)
    elif pooling_type == "max_3d":
        pooling_layer = MaxPooling3D(window_size=window_size, stride_size=stride_size,
            num_gpus=num_gpus, default_gpu_id=default_gpu_id, scope=scope)
    elif pooling_type == "avg":
        pooling_layer = AveragePooling(num_gpus=num_gpus, default_gpu_id=default_gpu_id, scope=scope)
    elif pooling_type == "avg_3d":
        pooling_layer = AveragePooling3D(window_size=window_size, stride_size=stride_size,
            num_gpus=num_gpus, default_gpu_id=default_gpu_id, scope=scope)
    else:
        raise ValueError("unsupported pooling type {0}".format(pooling_type))
    
    return pooling_layer

def create_dense_layer(dense_type,
                       num_layer,
                       unit_dim,
                       inner_scale,
                       activation,
                       dropout,
                       layer_dropout,
                       layer_norm,
                       residual_connect,
                       use_bias,
                       num_gpus,
                       default_gpu_id,
                       regularizer,
                       random_seed,
                       trainable):
    """create dense layer"""
    scope = "dense/{0}".format(dense_type)
    if dense_type == "single":
        dense_layer = StackedDense(num_layer=num_layer, unit_dim=unit_dim,
            activation=activation, dropout=dropout, layer_dropout=layer_dropout, layer_norm=layer_norm,
            residual_connect=residual_connect, use_bias=use_bias, num_gpus=num_gpus, default_gpu_id=default_gpu_id,
            regularizer=regularizer, random_seed=random_seed, trainable=trainable, scope=scope)
    elif dense_type == "double":
        dense_layer = StackedDoubleDense(num_layer=num_layer, unit_dim=unit_dim, inner_scale=inner_scale,
            activation=activation, dropout=dropout, layer_dropout=layer_dropout, layer_norm=layer_norm,
            residual_connect=residual_connect, use_bias=use_bias, num_gpus=num_gpus, default_gpu_id=default_gpu_id,
            regularizer=regularizer, random_seed=random_seed, trainable=trainable, scope=scope)
    else:
        raise ValueError("unsupported dense type {0}".format(dense_type))
    
    return dense_layer

def create_highway_layer(highway_type,
                         num_layer,
                         unit_dim,
                         window_size,
                         activation,
                         dropout,
                         num_gpus,
                         default_gpu_id,
                         regularizer,
                         random_seed,
                         trainable):
    """create highway layer"""
    scope = "highway/{0}".format(highway_type)
    if highway_type == "highway":
        highway_layer = StackedHighway(num_layer=num_layer, unit_dim=unit_dim,
            activation=activation, dropout=dropout, num_gpus=num_gpus, default_gpu_id=default_gpu_id,
            regularizer=regularizer, random_seed=random_seed, trainable=trainable)
    elif highway_type == "conv_highway":
        highway_layer = StackedHighway(num_layer=num_layer, num_filter=unit_dim, window_size=window_size,
            activation=activation, dropout=dropout, num_gpus=num_gpus, default_gpu_id=default_gpu_id,
            regularizer=regularizer, random_seed=random_seed, trainable=trainable)
    else:
        raise ValueError("unsupported dense type {0}".format(highway_type))
    
    return highway_layer

def create_recurrent_layer(recurrent_type,
                           num_layer,
                           unit_dim,
                           cell_type,
                           activation,
                           dropout,
                           forget_bias,
                           residual_connect,
                           attention_mechanism,
                           num_gpus,
                           default_gpu_id,
                           random_seed,
                           trainable):
    """create recurrent layer"""
    scope = "recurrent/{0}".format(recurrent_type)
    if recurrent_type == "uni":
        recurrent_layer = RNN(num_layer=num_layer, unit_dim=unit_dim, cell_type=cell_type,
            activation=activation, dropout=dropout, forget_bias=forget_bias, residual_connect=residual_connect,
            attention_mechanism=attention_mechanism, num_gpus=num_gpus, default_gpu_id=default_gpu_id,
            random_seed=random_seed, trainable=trainable, scope=scope)
    elif recurrent_type == "bi":
        recurrent_layer = BiRNN(num_layer=num_layer, unit_dim=unit_dim, cell_type=cell_type,
            activation=activation, dropout=dropout, forget_bias=forget_bias, residual_connect=residual_connect,
            attention_mechanism=attention_mechanism, num_gpus=num_gpus, default_gpu_id=default_gpu_id,
            random_seed=random_seed, trainable=trainable, scope=scope)
    else:
        raise ValueError("unsupported recurrent type {0}".format(recurrent_type))
    
    return recurrent_layer

def create_attention_layer(attention_type,
                           src_dim,
                           trg_dim,
                           att_dim,
                           num_head,
                           score_type,
                           dropout,
                           att_dropout,
                           layer_dropout,
                           layer_norm,
                           residual_connect,
                           is_self,
                           external_matrix,
                           num_gpus,
                           default_gpu_id,
                           regularizer,
                           random_seed,
                           trainable):
    """create attention layer"""
    scope = "attention/{0}".format(attention_type)
    if attention_type == "att":
        attention_layer = Attention(src_dim=src_dim, trg_dim=trg_dim, att_dim=att_dim,
            score_type=score_type, dropout=dropout, att_dropout=att_dropout, layer_dropout=layer_dropout,
            layer_norm=layer_norm, residual_connect=residual_connect, is_self=is_self,
            external_matrix=external_matrix, num_gpus=num_gpus, default_gpu_id=default_gpu_id,
            regularizer=regularizer, random_seed=random_seed, trainable=trainable, scope=scope)
    elif attention_type == "max_att":
        attention_layer = MaxAttention(src_dim=src_dim, trg_dim=trg_dim, att_dim=att_dim,
            score_type=score_type, dropout=dropout, att_dropout=att_dropout, layer_dropout=layer_dropout,
            layer_norm=layer_norm, residual_connect=residual_connect, is_self=is_self,
            external_matrix=external_matrix, num_gpus=num_gpus, default_gpu_id=default_gpu_id,
            regularizer=regularizer, random_seed=random_seed, trainable=trainable, scope=scope)
    elif attention_type == "co_att":
        attention_layer = CoAttention(src_dim=src_dim, trg_dim=trg_dim, att_dim=att_dim,
            score_type=score_type, dropout=dropout, att_dropout=att_dropout, layer_dropout=layer_dropout,
            layer_norm=layer_norm, residual_connect=residual_connect, is_self=is_self,
            external_matrix=external_matrix, num_gpus=num_gpus, default_gpu_id=default_gpu_id,
            regularizer=regularizer, random_seed=random_seed, trainable=trainable, scope=scope)
    elif attention_type == "gated_att":
        attention_layer = GatedAttention(src_dim=src_dim, trg_dim=trg_dim, att_dim=att_dim,
            score_type=score_type, dropout=dropout, att_dropout=att_dropout, layer_dropout=layer_dropout,
            layer_norm=layer_norm, residual_connect=residual_connect, is_self=is_self,
            external_matrix=external_matrix, num_gpus=num_gpus, default_gpu_id=default_gpu_id,
            regularizer=regularizer, random_seed=random_seed, trainable=trainable, scope=scope)
    elif attention_type == "multi_head_att":
        attention_layer = MultiHeadAttention(src_dim=src_dim, trg_dim=trg_dim, att_dim=att_dim, num_head=num_head,
            score_type=score_type, dropout=dropout, att_dropout=att_dropout, layer_dropout=layer_dropout,
            layer_norm=layer_norm, residual_connect=residual_connect, is_self=is_self,
            external_matrix=external_matrix, num_gpus=num_gpus, default_gpu_id=default_gpu_id, 
            regularizer=regularizer, random_seed=random_seed, trainable=trainable, scope=scope)
    else:
        raise ValueError("unsupported attention type {0}".format(attention_type))
    
    return attention_layer

class AttentionMechanism(object):
    def __init__(self,
                 memory,
                 memory_mask,
                 attention_type,
                 src_dim,
                 trg_dim,
                 att_dim,
                 score_type,
                 dropout,
                 att_dropout=0.0,
                 layer_dropout=0.0,
                 layer_norm=False,
                 residual_connect=False,
                 is_self=False,
                 external_matrix=None,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True):
        """initialize attention mechanism"""
        self.memory = memory
        self.memory_mask = memory_mask
        
        self.attention_layer = create_attention_layer(attention_type, src_dim, trg_dim, att_dim, -1,
            score_type, dropout, att_dropout, layer_dropout, layer_norm, residual_connect, is_self,
            external_matrix, num_gpus, default_gpu_id, False, regularizer, random_seed, trainable)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call attention mechanism"""
        output_attention, output_mask = self.attention_layer(input_data, self.memory, input_mask, self.memory_mask)
        return output_attention, output_mask

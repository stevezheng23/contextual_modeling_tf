import collections

import numpy as np
import tensorflow as tf

from model.dam import *
from util.data_util import *

__all__ = ["TrainModel", "InferModel",
           "create_train_model", "create_infer_model",
           "init_model", "load_model"]

class TrainModel(collections.namedtuple("TrainModel", ("graph", "model", "data_pipeline", "word_embedding"))):
    pass

class InferModel(collections.namedtuple("InferModel",
    ("graph", "model", "data_pipeline", "word_embedding", "input_data", "input_context", "input_response", "input_label"))):
    pass

def create_train_model(logger,
                       hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare train data")
        (input_data, input_context_data, input_response_data, input_label_data,
            word_embed_data, word_vocab_size, word_vocab_index, word_vocab_inverted_index,  
            char_vocab_size, char_vocab_index, char_vocab_inverted_index) = prepare_contextual_data(logger,
            hyperparams.data_train_contextual_file, hyperparams.data_train_contextual_file_type, hyperparams.data_word_vocab_file,
            hyperparams.data_word_vocab_size, hyperparams.data_word_vocab_threshold, hyperparams.model_representation_word_embed_dim,
            hyperparams.data_embedding_file, hyperparams.data_full_embedding_file, hyperparams.data_word_unk, hyperparams.data_word_pad,
            hyperparams.model_representation_word_feat_enable, hyperparams.model_representation_word_embed_pretrained,
            hyperparams.data_char_vocab_file, hyperparams.data_char_vocab_size, hyperparams.data_char_vocab_threshold,
            hyperparams.data_char_unk, hyperparams.data_char_pad, hyperparams.model_representation_char_feat_enable)
        
        logger.log_print("# create train context dataset")
        context_dataset = tf.data.Dataset.from_tensor_slices(input_context_data)
        input_context_word_dataset, input_context_char_dataset = create_src_dataset(context_dataset, True,
            hyperparams.data_context_utterance_size, word_vocab_index, hyperparams.data_context_word_size,
            hyperparams.data_word_pad, hyperparams.model_representation_word_feat_enable, char_vocab_index,
            hyperparams.data_context_char_size, hyperparams.data_char_pad, hyperparams.model_representation_char_feat_enable)
        
        logger.log_print("# create train response dataset")
        response_dataset = tf.data.Dataset.from_tensor_slices(input_response_data)
        input_response_word_dataset, input_response_char_dataset = create_src_dataset(response_dataset, False,
            hyperparams.data_response_candidate_size, word_vocab_index, hyperparams.data_response_word_size,
            hyperparams.data_word_pad, hyperparams.model_representation_word_feat_enable, char_vocab_index,
            hyperparams.data_response_char_size, hyperparams.data_char_pad, hyperparams.model_representation_char_feat_enable)
        
        logger.log_print("# create train label dataset")
        label_dataset = tf.data.Dataset.from_tensor_slices(input_label_data)
        input_label_dataset = create_trg_dataset(label_dataset, False, hyperparams.data_response_candidate_size)
        
        logger.log_print("# create train data pipeline")
        data_pipeline = create_data_pipeline(input_context_word_dataset, input_context_char_dataset,
            input_response_word_dataset, input_response_char_dataset, input_label_dataset, word_vocab_index,
            hyperparams.data_word_pad, hyperparams.model_representation_word_feat_enable, char_vocab_index,
            hyperparams.data_char_pad, hyperparams.model_representation_char_feat_enable,
            hyperparams.train_enable_shuffle, hyperparams.train_shuffle_buffer_size,
            len(input_data), hyperparams.train_batch_size, hyperparams.train_random_seed)
        
        model_creator = get_model_creator(hyperparams.model_type)
        model = model_creator(logger=logger, hyperparams=hyperparams, data_pipeline=data_pipeline,
            mode="train", scope=hyperparams.model_scope)
        
        return TrainModel(graph=graph, model=model, data_pipeline=data_pipeline, word_embedding=word_embed_data)

def create_infer_model(logger,
                       hyperparams):
    graph = tf.Graph()
    with graph.as_default():
        logger.log_print("# prepare infer data")
        (input_data, input_context_data, input_response_data, input_label_data,
             word_embed_data, word_vocab_size, word_vocab_index, word_vocab_inverted_index,  
             char_vocab_size, char_vocab_index, char_vocab_inverted_index) = prepare_contextual_data(logger,
             hyperparams.data_eval_contextual_file, hyperparams.data_eval_contextual_file_type, hyperparams.data_word_vocab_file,
             hyperparams.data_word_vocab_size, hyperparams.data_word_vocab_threshold, hyperparams.model_representation_word_embed_dim,
             hyperparams.data_embedding_file, hyperparams.data_full_embedding_file, hyperparams.data_word_unk, hyperparams.data_word_pad,
             hyperparams.model_representation_word_feat_enable, hyperparams.model_representation_word_embed_pretrained,
             hyperparams.data_char_vocab_file, hyperparams.data_char_vocab_size, hyperparams.data_char_vocab_threshold,
             hyperparams.data_char_unk, hyperparams.data_char_pad, hyperparams.model_representation_char_feat_enable)
        
        logger.log_print("# create infer context dataset")
        context_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        context_dataset = tf.data.Dataset.from_tensor_slices(context_placeholder)
        input_context_word_dataset, input_context_char_dataset = create_src_dataset(context_dataset, True,
             hyperparams.data_context_utterance_size, word_vocab_index, hyperparams.data_context_word_size,
             hyperparams.data_word_pad, hyperparams.model_representation_word_feat_enable, char_vocab_index,
             hyperparams.data_context_char_size, hyperparams.data_char_pad, hyperparams.model_representation_char_feat_enable)
        
        logger.log_print("# create infer response dataset")
        response_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        response_dataset = tf.data.Dataset.from_tensor_slices(response_placeholder)
        input_response_word_dataset, input_response_char_dataset = create_src_dataset(response_dataset, False,
             hyperparams.data_response_candidate_size, word_vocab_index, hyperparams.data_response_word_size,
             hyperparams.data_word_pad, hyperparams.model_representation_word_feat_enable, char_vocab_index,
             hyperparams.data_response_char_size, hyperparams.data_char_pad, hyperparams.model_representation_char_feat_enable)
        
        logger.log_print("# create infer label dataset")
        label_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        label_dataset = tf.data.Dataset.from_tensor_slices(label_placeholder)
        input_label_dataset = create_trg_dataset(label_dataset, False, hyperparams.data_response_candidate_size)
        
        logger.log_print("# create infer data pipeline")
        data_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
        batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
        data_pipeline = create_dynamic_pipeline(input_context_word_dataset, input_context_char_dataset,
            input_response_word_dataset, input_response_char_dataset, input_label_dataset, word_vocab_index,
            hyperparams.data_word_pad, hyperparams.model_representation_word_feat_enable, char_vocab_index,
            hyperparams.data_char_pad, hyperparams.model_representation_char_feat_enable, context_placeholder,
            response_placeholder, label_placeholder, data_size_placeholder, batch_size_placeholder)
        
        model_creator = get_model_creator(hyperparams.model_type)
        model = model_creator(logger=logger, hyperparams=hyperparams, data_pipeline=data_pipeline,
            mode="infer", scope=hyperparams.model_scope)
        
        return InferModel(graph=graph, model=model, data_pipeline=data_pipeline,
            word_embedding=word_embed_data, input_data=input_data, input_context=input_context_data,
            input_response=input_response_data, input_label=input_label_data)

def get_model_creator(model_type):
    if model_type == "dam":
        model_creator = DAM
    else:
        raise ValueError("can not create model with unsupported model type {0}".format(model_type))
    
    return model_creator

def init_model(sess,
               model):
    with model.graph.as_default():
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

def load_model(sess,
               model,
               ckpt_file,
               ckpt_type):
    with model.graph.as_default():
        model.model.restore(sess, ckpt_file, ckpt_type)

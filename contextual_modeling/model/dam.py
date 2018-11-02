import collections
import functools
import os.path
import operator

import numpy as np
import tensorflow as tf

from functools import reduce

from util.default_util import *
from util.contextual_modeling_util import *
from util.layer_util import *

from model.base_model import *

__all__ = ["DAM"]

class DAM(BaseModel):
    """deep attention matching (dam) model"""
    def __init__(self,
                 logger,
                 hyperparams,
                 data_pipeline,
                 mode="train",
                 scope="dam"):
        """initialize dam model"""
        super(DAM, self).__init__(logger=logger, hyperparams=hyperparams,
            data_pipeline=data_pipeline, mode=mode, scope=scope)
        
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.global_step = tf.get_variable("global_step", shape=[], dtype=tf.int32,
                initializer=tf.zeros_initializer, trainable=False)
                        
            """get batch input from data pipeline"""
            context_word = self.data_pipeline.input_context_word
            context_word_mask = self.data_pipeline.input_context_word_mask
            context_char = self.data_pipeline.input_context_char
            context_char_mask = self.data_pipeline.input_context_char_mask
            response_word = self.data_pipeline.input_response_word
            response_word_mask = self.data_pipeline.input_response_word_mask
            response_char = self.data_pipeline.input_response_char
            response_char_mask = self.data_pipeline.input_response_char_mask
            label = self.data_pipeline.input_label
            label_mask = self.data_pipeline.input_label_mask
            
            """build graph for dam model"""
            self.logger.log_print("# build graph")
            predict, predict_mask = self._build_graph(context_word, context_char, response_word, response_char,
                context_word_mask, context_char_mask, response_word_mask, response_char_mask)
            self.predict = predict
            self.predict_mask = predict_mask
            
            self.variable_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            self.variable_lookup = {v.op.name: v for v in self.variable_list}
            
            if self.hyperparams.train_ema_enable == True:
                self.ema = tf.train.ExponentialMovingAverage(decay=self.hyperparams.train_ema_decay_rate)
                self.variable_lookup = {self.ema.average_name(v): v for v in self.variable_list}
            
            if self.mode == "infer":
                """get infer answer"""
                self.infer_predict = tf.nn.sigmoid(self.predict)
                self.infer_predict_mask = self.predict_mask
                
                """create infer summary"""
                self.infer_summary = self._get_infer_summary()
            
            if self.mode == "train":
                """compute optimization loss"""
                self.logger.log_print("# setup loss computation mechanism")
                self.train_loss = self._compute_loss(label, label_mask, self.predict, self.predict_mask)
                
                if self.hyperparams.train_regularization_enable == True:
                    regularization_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                    regularization_loss = tf.contrib.layers.apply_regularization(self.regularizer, regularization_variables)
                    self.train_loss = self.train_loss + regularization_loss
                
                """apply learning rate warm-up & decay"""
                self.logger.log_print("# setup initial learning rate mechanism")
                self.initial_learning_rate = tf.constant(self.hyperparams.train_optimizer_learning_rate)
                
                if self.hyperparams.train_optimizer_warmup_enable == True:
                    self.logger.log_print("# setup learning rate warm-up mechanism")
                    self.warmup_learning_rate = self._apply_learning_rate_warmup(self.initial_learning_rate)
                else:
                    self.warmup_learning_rate = self.initial_learning_rate
                
                if self.hyperparams.train_optimizer_decay_enable == True:
                    self.logger.log_print("# setup learning rate decay mechanism")
                    self.decayed_learning_rate = self._apply_learning_rate_decay(self.warmup_learning_rate)
                else:
                    self.decayed_learning_rate = self.warmup_learning_rate
                
                self.learning_rate = self.decayed_learning_rate
                
                """initialize optimizer"""
                self.logger.log_print("# setup training optimizer")
                self.optimizer = self._initialize_optimizer(self.learning_rate)
                
                """minimize optimization loss"""
                self.logger.log_print("# setup loss minimization mechanism")
                self.update_model, self.clipped_gradients, self.gradient_norm = self._minimize_loss(self.train_loss)
                
                if self.hyperparams.train_ema_enable == True:
                    with tf.control_dependencies([self.update_model]):
                        self.update_op = self.ema.apply(self.variable_list)
                        self.variable_lookup = {self.ema.average_name(v): self.ema.average(v) for v in self.variable_list}
                else:
                    self.update_op = self.update_model
                
                """create train summary"""
                self.train_summary = self._get_train_summary()
            
            """create checkpoint saver"""
            if not tf.gfile.Exists(self.hyperparams.train_ckpt_output_dir):
                tf.gfile.MakeDirs(self.hyperparams.train_ckpt_output_dir)
            
            self.ckpt_debug_dir = os.path.join(self.hyperparams.train_ckpt_output_dir, "debug")
            self.ckpt_epoch_dir = os.path.join(self.hyperparams.train_ckpt_output_dir, "epoch")
            
            if not tf.gfile.Exists(self.ckpt_debug_dir):
                tf.gfile.MakeDirs(self.ckpt_debug_dir)
            
            if not tf.gfile.Exists(self.ckpt_epoch_dir):
                tf.gfile.MakeDirs(self.ckpt_epoch_dir)
            
            self.ckpt_debug_name = os.path.join(self.ckpt_debug_dir, "model_debug_ckpt")
            self.ckpt_epoch_name = os.path.join(self.ckpt_epoch_dir, "model_epoch_ckpt")
            self.ckpt_debug_saver = tf.train.Saver(self.variable_lookup)
            self.ckpt_epoch_saver = tf.train.Saver(self.variable_lookup, max_to_keep=self.hyperparams.train_num_epoch)
    
    def _build_representation_layer(self,
                                    input_context_word,
                                    input_context_word_mask,
                                    input_context_char,
                                    input_context_char_mask,
                                    input_response_word,
                                    input_response_word_mask,
                                    input_response_char,
                                    input_response_char_mask):
        """build representation layer for dam model"""
        word_vocab_size = self.hyperparams.data_word_vocab_size
        word_embed_dim = self.hyperparams.model_representation_word_embed_dim
        word_dropout = self.hyperparams.model_representation_word_dropout if self.mode == "train" else 0.0
        word_embed_pretrained = self.hyperparams.model_representation_word_embed_pretrained
        word_feat_trainable = self.hyperparams.model_representation_word_feat_trainable
        word_feat_enable = self.hyperparams.model_representation_word_feat_enable
        char_vocab_size = self.hyperparams.data_char_vocab_size
        char_embed_dim = self.hyperparams.model_representation_char_embed_dim
        char_unit_dim = self.hyperparams.model_representation_char_unit_dim
        char_window_size = self.hyperparams.model_representation_char_window_size
        char_hidden_activation = self.hyperparams.model_representation_char_hidden_activation
        char_dropout = self.hyperparams.model_representation_char_dropout if self.mode == "train" else 0.0
        char_pooling_type = self.hyperparams.model_representation_char_pooling_type
        char_feat_trainable = self.hyperparams.model_representation_char_feat_trainable
        char_feat_enable = self.hyperparams.model_representation_char_feat_enable
        fusion_type = self.hyperparams.model_representation_fusion_type
        fusion_num_layer = self.hyperparams.model_representation_fusion_num_layer
        fusion_unit_dim = self.hyperparams.model_representation_fusion_unit_dim
        fusion_hidden_activation = self.hyperparams.model_representation_fusion_hidden_activation
        fusion_dropout = self.hyperparams.model_representation_fusion_dropout if self.mode == "train" else 0.0
        fusion_trainable = self.hyperparams.model_representation_fusion_trainable
        random_seed = self.hyperparams.train_random_seed
        default_representation_gpu_id = self.default_gpu_id
        
        with tf.variable_scope("representation", reuse=tf.AUTO_REUSE):
            input_context_feat_list = []
            input_context_feat_mask_list = []
            input_response_feat_list = []
            input_response_feat_mask_list = []
            
            if word_feat_enable == True:
                self.logger.log_print("# build word-level representation layer")
                word_feat_layer = WordFeat(vocab_size=word_vocab_size, embed_dim=word_embed_dim,
                    dropout=word_dropout, pretrained=word_embed_pretrained, random_seed=random_seed, trainable=word_feat_trainable)
                
                (input_context_word_feat,
                    input_context_word_feat_mask) = word_feat_layer(input_context_word, input_context_word_mask)
                (input_response_word_feat,
                    input_response_word_feat_mask) = word_feat_layer(input_response_word, input_response_word_mask)
                input_context_feat_list.append(input_context_word_feat)
                input_context_feat_mask_list.append(input_context_word_feat_mask)
                input_response_feat_list.append(input_response_word_feat)
                input_response_feat_mask_list.append(input_response_word_feat_mask)
                
                word_unit_dim = word_embed_dim
                self.word_embedding_placeholder = word_feat_layer.get_embedding_placeholder()
            else:
                word_unit_dim = 0
                self.word_embedding_placeholder = None
            
            if char_feat_enable == True:
                self.logger.log_print("# build char-level representation layer")
                char_feat_layer = CharFeat(vocab_size=char_vocab_size, embed_dim=char_embed_dim, unit_dim=char_unit_dim,
                    window_size=char_window_size, activation=char_hidden_activation, pooling_type=char_pooling_type,
                    dropout=char_dropout, num_gpus=self.num_gpus, default_gpu_id=default_representation_gpu_id,
                    regularizer=self.regularizer, random_seed=random_seed, trainable=char_feat_trainable)
                
                (input_context_char_feat,
                    input_context_char_feat_mask) = char_feat_layer(input_context_char, input_context_char_mask)
                (input_response_char_feat,
                    input_response_char_feat_mask) = char_feat_layer(input_response_char, input_response_char_mask)
                
                input_context_feat_list.append(input_context_char_feat)
                input_context_feat_mask_list.append(input_context_char_feat_mask)
                input_response_feat_list.append(input_response_char_feat)
                input_response_feat_mask_list.append(input_response_char_feat_mask)
            else:
                char_unit_dim = 0
            
            feat_unit_dim = word_unit_dim + char_unit_dim
            feat_fusion_layer = FusionModule(input_unit_dim=feat_unit_dim, output_unit_dim=fusion_unit_dim,
                fusion_type=fusion_type, num_layer=fusion_num_layer, activation=fusion_hidden_activation,
                dropout=fusion_dropout, num_gpus=self.num_gpus, default_gpu_id=default_representation_gpu_id,
                regularizer=self.regularizer, random_seed=random_seed, trainable=fusion_trainable)
            
            (input_context_feat,
                input_context_feat_mask) = feat_fusion_layer(input_context_feat_list, input_context_feat_mask_list)
            (input_response_feat,
                input_response_feat_mask) = feat_fusion_layer(input_response_feat_list, input_response_feat_mask_list)
            self.input_context_feat_mask = input_context_feat_mask
            self.input_response_feat_mask = input_response_feat_mask
        
        return input_context_feat, input_response_feat, input_context_feat_mask, input_response_feat_mask
    
    def _build_understanding_layer(self,
                                   context_feat,
                                   response_feat,
                                   context_feat_mask,
                                   response_feat_mask):
        """build understanding layer for dam model"""
        context_utterance_size = self.hyperparams.data_context_utterance_size
        response_candidate_size = self.hyperparams.data_response_candidate_size
        context_representation_unit_dim = self.hyperparams.model_representation_fusion_unit_dim
        response_representation_unit_dim = self.hyperparams.model_representation_fusion_unit_dim
        context_understanding_num_layer = self.hyperparams.model_understanding_context_num_layer
        context_understanding_num_head = self.hyperparams.model_understanding_context_num_head
        context_understanding_unit_dim = self.hyperparams.model_understanding_context_unit_dim
        context_understanding_hidden_activation = self.hyperparams.model_understanding_context_hidden_activation
        context_understanding_dropout = self.hyperparams.model_understanding_context_dropout if self.mode == "train" else 0.0
        context_understanding_layer_dropout = self.hyperparams.model_understanding_context_layer_dropout if self.mode == "train" else 0.0
        context_understanding_trainable = self.hyperparams.model_understanding_context_trainable
        response_understanding_num_layer = self.hyperparams.model_understanding_response_num_layer
        response_understanding_num_head = self.hyperparams.model_understanding_response_num_head
        response_understanding_unit_dim = self.hyperparams.model_understanding_response_unit_dim
        response_understanding_hidden_activation = self.hyperparams.model_understanding_response_hidden_activation
        response_understanding_dropout = self.hyperparams.model_understanding_response_dropout if self.mode == "train" else 0.0
        response_understanding_layer_dropout = self.hyperparams.model_understanding_response_layer_dropout if self.mode == "train" else 0.0
        response_understanding_trainable = self.hyperparams.model_understanding_response_trainable
        enable_understanding_sharing = self.hyperparams.model_understanding_enable_sharing
        random_seed = self.hyperparams.train_random_seed
        default_understanding_gpu_id = self.default_gpu_id
        
        with tf.variable_scope("understanding", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("context", reuse=tf.AUTO_REUSE):
                self.logger.log_print("# build context understanding layer")
                context_understanding_fusion_layer = FusionModule(input_unit_dim=context_representation_unit_dim,
                    output_unit_dim=context_understanding_unit_dim, fusion_type="conv", num_layer=1,
                    activation=context_understanding_hidden_activation, dropout=context_understanding_dropout,
                    num_gpus=self.num_gpus, default_gpu_id=default_understanding_gpu_id, regularizer=self.regularizer,
                    random_seed=random_seed, trainable=context_understanding_trainable)
                context_understanding_layer = StackedAttentiveModule(num_layer=context_understanding_num_layer,
                    num_head=context_understanding_num_head, unit_dim=context_understanding_unit_dim,
                    activation=context_understanding_hidden_activation, dropout=context_understanding_dropout,
                    layer_dropout=context_understanding_layer_dropout, num_gpus=self.num_gpus,
                    default_gpu_id=default_understanding_gpu_id, regularizer=self.regularizer,
                    random_seed=random_seed, trainable=context_understanding_trainable)
                
                (context_understanding_fusion,
                    context_understanding_fusion_mask) = context_understanding_fusion_layer([context_feat], [context_feat_mask])
                (context_understanding,
                    context_understanding_mask) = context_understanding_layer(context_understanding_fusion,
                        context_understanding_fusion_mask)
                
                context_understanding = [tf.tile(tf.expand_dims(c, axis=1),
                    multiples=[1, response_candidate_size, 1, 1, 1]) for c in context_understanding]
                context_understanding_mask = [tf.tile(tf.expand_dims(c_mask, axis=1),
                    multiples=[1, response_candidate_size, 1, 1, 1]) for c_mask in context_understanding_mask]
            
            with tf.variable_scope("response", reuse=tf.AUTO_REUSE):
                self.logger.log_print("# build response understanding layer")
                if (enable_understanding_sharing == True and context_representation_unit_dim == response_representation_unit_dim and
                    context_understanding_unit_dim == response_understanding_unit_dim):
                    response_understanding_fusion_layer = context_understanding_fusion_layer
                    response_understanding_layer = context_understanding_layer
                else:
                    response_understanding_fusion_layer = FusionModule(input_unit_dim=response_representation_unit_dim,
                        output_unit_dim=response_understanding_unit_dim, fusion_type="conv", num_layer=1,
                        activation=response_understanding_hidden_activation, dropout=response_understanding_dropout,
                        num_gpus=self.num_gpus, default_gpu_id=default_understanding_gpu_id, regularizer=self.regularizer,
                        random_seed=random_seed, trainable=response_understanding_trainable)
                    response_understanding_layer = StackedAttentiveModule(num_layer=response_understanding_num_layer,
                        num_head=response_understanding_num_head, unit_dim=response_understanding_unit_dim,
                        activation=response_understanding_hidden_activation, dropout=response_understanding_dropout,
                        layer_dropout=response_understanding_layer_dropout, num_gpus=self.num_gpus,
                        default_gpu_id=default_understanding_gpu_id,regularizer=self.regularizer,
                        random_seed=random_seed, trainable=response_understanding_trainable)
                
                (response_understanding_fusion,
                    response_understanding_fusion_mask) = response_understanding_fusion_layer([response_feat], [response_feat_mask])
                (response_understanding,
                    response_understanding_mask) = response_understanding_layer(response_understanding_fusion,
                        response_understanding_fusion_mask)
                
                response_understanding = [tf.tile(tf.expand_dims(r, axis=2),
                    multiples=[1, 1, context_utterance_size, 1, 1]) for r in response_understanding]
                response_understanding_mask = [tf.tile(tf.expand_dims(r_mask, axis=2),
                    multiples=[1, 1, context_utterance_size, 1, 1]) for r_mask in response_understanding_mask]
        
        return context_understanding, response_understanding, context_understanding_mask, response_understanding_mask
    
    def _build_interaction_layer(self,
                                 context_understanding,
                                 response_understanding,
                                 context_understanding_mask,
                                 response_understanding_mask):
        """build interaction layer for dam model"""
        context_understanding_unit_dim = self.hyperparams.model_understanding_context_unit_dim
        response_understanding_unit_dim = self.hyperparams.model_understanding_response_unit_dim
        context2response_interaction_num_layer = self.hyperparams.model_interaction_context2response_num_layer
        context2response_interaction_num_head = self.hyperparams.model_interaction_context2response_num_head
        context2response_interaction_unit_dim = self.hyperparams.model_interaction_context2response_unit_dim
        context2response_interaction_hidden_activation = self.hyperparams.model_interaction_context2response_hidden_activation
        context2response_interaction_dropout = self.hyperparams.model_interaction_context2response_dropout if self.mode == "train" else 0.0
        context2response_interaction_layer_dropout = self.hyperparams.model_interaction_context2response_layer_dropout if self.mode == "train" else 0.0
        context2response_interaction_trainable = self.hyperparams.model_interaction_context2response_trainable
        response2context_interaction_num_layer = self.hyperparams.model_interaction_response2context_num_layer
        response2context_interaction_num_head = self.hyperparams.model_interaction_response2context_num_head
        response2context_interaction_unit_dim = self.hyperparams.model_interaction_response2context_unit_dim
        response2context_interaction_hidden_activation = self.hyperparams.model_interaction_response2context_hidden_activation
        response2context_interaction_dropout = self.hyperparams.model_interaction_response2context_dropout if self.mode == "train" else 0.0
        response2context_interaction_layer_dropout = self.hyperparams.model_interaction_response2context_layer_dropout if self.mode == "train" else 0.0
        response2context_interaction_trainable = self.hyperparams.model_interaction_response2context_trainable
        random_seed = self.hyperparams.train_random_seed
        default_interaction_gpu_id = self.default_gpu_id + 1
        
        with tf.variable_scope("interaction", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("context2response", reuse=tf.AUTO_REUSE):
                self.logger.log_print("# build context-to-response interaction layer")
                context2response_interaction_layer = MultiAttentiveModule(num_layer=context2response_interaction_num_layer,
                    num_head=context2response_interaction_num_head, unit_dim=context2response_interaction_unit_dim,
                    activation=context2response_interaction_hidden_activation, dropout=context2response_interaction_dropout,
                    layer_dropout=context2response_interaction_layer_dropout, num_gpus=self.num_gpus,
                    default_gpu_id=default_interaction_gpu_id, regularizer=self.regularizer,
                    random_seed=random_seed, trainable=context2response_interaction_trainable)
                
                (context2response_interaction,
                    context2response_interaction_mask) = context2response_interaction_layer(context_understanding,
                    response_understanding, context_understanding_mask, response_understanding_mask)
            
            with tf.variable_scope("response2context", reuse=tf.AUTO_REUSE):
                self.logger.log_print("# build response-to-context interaction layer")
                response2context_interaction_layer = MultiAttentiveModule(num_layer=response2context_interaction_num_layer,
                    num_head=response2context_interaction_num_head, unit_dim=response2context_interaction_unit_dim,
                    activation=response2context_interaction_hidden_activation, dropout=response2context_interaction_dropout,
                    layer_dropout=response2context_interaction_layer_dropout, num_gpus=self.num_gpus,
                    default_gpu_id=default_interaction_gpu_id, regularizer=self.regularizer,
                    random_seed=random_seed, trainable=response2context_interaction_trainable)
                
                (response2context_interaction,
                    response2context_interaction_mask) = response2context_interaction_layer(response_understanding,
                    context_understanding, response_understanding_mask, context_understanding_mask)
        
        return (context2response_interaction, response2context_interaction,
            context2response_interaction_mask, response2context_interaction_mask)
    
    def _build_matching_layer(self,
                              context_understanding,
                              response_understanding,
                              context2response_interaction,
                              response2context_interaction,
                              context_understanding_mask,
                              response_understanding_mask,
                              context2response_interaction_mask,
                              response2context_interaction_mask):
        """build matching layer for dam model"""
        aggregation_num_layer = self.hyperparams.model_matching_aggregation_num_layer
        aggregation_unit_dim = self.hyperparams.model_matching_aggregation_unit_dim
        aggregation_hidden_activation = self.hyperparams.model_matching_aggregation_hidden_activation
        aggregation_conv_window = self.hyperparams.model_matching_aggregation_conv_window
        aggregation_conv_stride = self.hyperparams.model_matching_aggregation_conv_stride
        aggregation_pool_window = self.hyperparams.model_matching_aggregation_pool_window
        aggregation_pool_stride = self.hyperparams.model_matching_aggregation_pool_stride
        aggregation_pooling_type = self.hyperparams.model_matching_aggregation_pooling_type
        aggregation_dropout = self.hyperparams.model_matching_aggregation_dropout
        aggregation_trainable = self.hyperparams.model_matching_aggregation_trainable
        projection_dropout = self.hyperparams.model_matching_projection_dropout
        projection_trainable = self.hyperparams.model_matching_projection_trainable
        random_seed = self.hyperparams.train_random_seed
        default_matching_gpu_id = self.default_gpu_id + 2
        
        with tf.variable_scope("matching", reuse=tf.AUTO_REUSE):
            self.logger.log_print("# build context-response matching layer")
            if (len(context_understanding) != len(response_understanding) or
                len(context2response_interaction) != len(response2context_interaction) or
                len(context_understanding_mask) != len(response_understanding_mask) or
                len(context2response_interaction_mask) != len(response2context_interaction_mask)):
                raise ValueError("number result of context & response and context2response & response2context must be the same")
            
            context_understanding_list = [tf.expand_dims(c, axis=3) for c in context_understanding]
            response_understanding_list = [tf.expand_dims(r, axis=3) for r in response_understanding]
            context2response_interaction_list = [tf.expand_dims(c2r, axis=3) for c2r in context2response_interaction]
            response2context_interaction_list = [tf.expand_dims(r2c, axis=3) for r2c in response2context_interaction]
            context_understanding_mask_list = [tf.expand_dims(c_mask, axis=3) for c_mask in context_understanding_mask]
            response_understanding_mask_list = [tf.expand_dims(r_mask, axis=3) for r_mask in response_understanding_mask]
            context2response_interaction_mask_list = [tf.expand_dims(c2r_mask, axis=3) for c2r_mask in context2response_interaction_mask]
            response2context_interaction_mask_list = [tf.expand_dims(r2c_mask, axis=3) for r2c_mask in response2context_interaction_mask]
            
            context_understanding = tf.concat(context_understanding_list, axis=3)
            response_understanding = tf.concat(response_understanding_list, axis=3)
            context2response_interaction = tf.concat(context2response_interaction_list, axis=3)
            response2context_interaction = tf.concat(response2context_interaction_list, axis=3)
            context_understanding_mask = tf.concat(context_understanding_mask_list, axis=3)
            response_understanding_mask = tf.concat(response_understanding_mask_list, axis=3)
            context2response_interaction_mask = tf.concat(context2response_interaction_mask_list, axis=3)
            response2context_interaction_mask = tf.concat(response2context_interaction_mask_list, axis=3)
            
            self_matching = tf.matmul(context_understanding, response_understanding, transpose_b=True)
            cross_matching = tf.matmul(context2response_interaction, response2context_interaction, transpose_b=True)
            self_matching_mask = tf.matmul(context_understanding_mask, response_understanding_mask, transpose_b=True)
            cross_matching_mask = tf.matmul(context2response_interaction_mask, response2context_interaction_mask, transpose_b=True)
            
            self_matching = tf.transpose(self_matching, perm=[0, 1, 2, 4, 5, 3])
            cross_matching = tf.transpose(cross_matching, perm=[0, 1, 2, 4, 5, 3])
            self_matching_mask = tf.reduce_max(tf.transpose(self_matching_mask, perm=[0, 1, 2, 4, 5, 3]), axis=-1, keepdims=True)
            cross_matching_mask = tf.reduce_max(tf.transpose(cross_matching_mask, perm=[0, 1, 2, 4, 5, 3]), axis=-1, keepdims=True)
            
            full_matching = tf.concat([self_matching, cross_matching], axis=-1)
            full_matching_mask = tf.reduce_max(tf.concat([self_matching_mask, cross_matching_mask], axis=-1), axis=-1, keepdims=True)
            
            full_matching_unit_dim = len(context_understanding_list) + len(context2response_interaction_list)
            aggregation_unit_dim = [full_matching_unit_dim] + aggregation_unit_dim
            aggregation_layer = StackedAggregationModule(num_layer=aggregation_num_layer, unit_dim=aggregation_unit_dim,
                activation=aggregation_hidden_activation, conv_window=aggregation_conv_window, conv_stride=aggregation_conv_stride,
                pool_window=aggregation_pool_window, pool_stride=aggregation_pool_stride, pooling_type=aggregation_pooling_type,
                dropout=aggregation_dropout, num_gpus=self.num_gpus, default_gpu_id=default_matching_gpu_id,
                regularizer=self.regularizer, random_seed=random_seed, trainable=aggregation_trainable)
            
            aggregated_matching, aggregated_matching_mask = aggregation_layer(full_matching, full_matching_mask)
            
            self.context_understanding_mask = context_understanding_mask
            self.response_understanding_mask = response_understanding_mask
            self.context2response_interaction_mask = context2response_interaction_mask
            self.response2context_interaction_mask = response2context_interaction_mask
            self.full_matching_mask = full_matching_mask
            self.aggregated_matching_mask = aggregated_matching_mask
            
            aggregated_matching_shape_prefix = tf.shape(aggregated_matching)[:2]
            aggregated_matching_shape_suffix = [reduce(operator.mul, aggregated_matching.get_shape().as_list()[2:])]
            aggregated_matching_shape = tf.concat([aggregated_matching_shape_prefix, aggregated_matching_shape_suffix], axis=0)
            aggregated_matching = tf.reshape(aggregated_matching, shape=aggregated_matching_shape)
            aggregated_matching_mask = tf.expand_dims(tf.reduce_max(aggregated_matching_mask, axis=[2, 3, 4, 5]), axis=-1)
            
            projection_layer = create_dense_layer("single", 1, 1, 1, "", [projection_dropout], None, False, False, 
                self.num_gpus, default_matching_gpu_id, self.regularizer, random_seed, projection_trainable)
            
            projection_matching, projection_matching_mask = projection_layer(aggregated_matching, aggregated_matching_mask)
            
            output_matching = projection_matching
            output_matching_mask = projection_matching_mask
        
        return output_matching, output_matching_mask
    
    def _build_graph(self,
                     context_word,
                     context_char,
                     response_word,
                     response_char,
                     context_word_mask,
                     context_char_mask,
                     response_word_mask,
                     response_char_mask):
        """build graph for dam model"""
        with tf.variable_scope("graph", reuse=tf.AUTO_REUSE):
            """build representation layer for dam model"""
            (context_feat, response_feat, context_feat_mask,
                response_feat_mask) = self._build_representation_layer(context_word, context_word_mask,
                context_char, context_char_mask, response_word, response_word_mask, response_char, response_char_mask)
            
            """build understanding layer for dam model"""
            (context_understanding, response_understanding, context_understanding_mask,
                response_understanding_mask) = self._build_understanding_layer(context_feat,
                response_feat, context_feat_mask, response_feat_mask)
            
            """build interaction layer for dam model"""
            (context2response_interaction, response2context_interaction, context2response_interaction_mask,
                response2context_interaction_mask) = self._build_interaction_layer(context_understanding,
                response_understanding, context_understanding_mask, response_understanding_mask)
            
            """build matching layer for dam model"""
            context_response_matching, context_response_matching_mask = self._build_matching_layer(context_understanding,
                response_understanding, context2response_interaction, response2context_interaction, context_understanding_mask,
                response_understanding_mask, context2response_interaction_mask, response2context_interaction_mask)
            
            predict = context_response_matching
            predict_mask = context_response_matching_mask
        
        return predict, predict_mask
    
    def _compute_loss(self,
                      label,
                      label_mask,
                      predict,
                      predict_mask):
        """compute optimization loss"""
        masked_label = label * label_mask
        masked_predict = predict * predict_mask
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=masked_predict, labels=masked_label)
        cross_entropy_mask = tf.reduce_max(tf.concat([label_mask, predict_mask], axis=-1), axis=-1, keepdims=True)
        masked_cross_entropy = cross_entropy * cross_entropy_mask
        loss = tf.reduce_mean(tf.reduce_sum(masked_cross_entropy, axis=-2))
        
        return loss
    
    def save(self,
             sess,
             global_step,
             save_mode):
        """save checkpoint for dam model"""
        if save_mode == "debug":
            self.ckpt_debug_saver.save(sess, self.ckpt_debug_name, global_step=global_step)
        elif save_mode == "epoch":
            self.ckpt_epoch_saver.save(sess, self.ckpt_epoch_name, global_step=global_step)
        else:
            raise ValueError("unsupported save mode {0}".format(save_mode))
    
    def restore(self,
                sess,
                ckpt_file,
                ckpt_type):
        """restore dam model from checkpoint"""
        if ckpt_file is None:
            raise FileNotFoundError("checkpoint file doesn't exist")
        
        if ckpt_type == "debug":
            self.ckpt_debug_saver.restore(sess, ckpt_file)
        elif ckpt_type == "epoch":
            self.ckpt_epoch_saver.restore(sess, ckpt_file)
        else:
            raise ValueError("unsupported checkpoint type {0}".format(ckpt_type))
    
    def get_latest_ckpt(self,
                        ckpt_type):
        """get the latest checkpoint for dam model"""
        if ckpt_type == "debug":
            ckpt_file = tf.train.latest_checkpoint(self.ckpt_debug_dir)
            if ckpt_file is None:
                raise FileNotFoundError("latest checkpoint file doesn't exist")
            
            return ckpt_file
        elif ckpt_type == "epoch":
            ckpt_file = tf.train.latest_checkpoint(self.ckpt_epoch_dir)
            if ckpt_file is None:
                raise FileNotFoundError("latest checkpoint file doesn't exist")
            
            return ckpt_file
        else:
            raise ValueError("unsupported checkpoint type {0}".format(ckpt_type))
    
    def get_ckpt_list(self,
                      ckpt_type):
        """get checkpoint list for dam model"""
        if ckpt_type == "debug":
            ckpt_state = tf.train.get_checkpoint_state(self.ckpt_debug_dir)
            if ckpt_state is None:
                raise FileNotFoundError("checkpoint files doesn't exist")
            
            return ckpt_state.all_model_checkpoint_paths
        elif ckpt_type == "epoch":
            ckpt_state = tf.train.get_checkpoint_state(self.ckpt_epoch_dir)
            if ckpt_state is None:
                raise FileNotFoundError("checkpoint files doesn't exist")
            
            return ckpt_state.all_model_checkpoint_paths
        else:
            raise ValueError("unsupported checkpoint type {0}".format(ckpt_type))

class AttentiveModule(object):
    """attentive-module layer"""
    def __init__(self,
                 num_head,
                 unit_dim,
                 activation,
                 dropout,
                 layer_dropout,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="att_module"):
        """initialize attentive-module layer"""
        self.num_head = num_head
        self.unit_dim = unit_dim
        self.activation = activation
        self.enable_dropout, self.dropout = dropout
        self.sublayer_skip, self.num_sublayer, self.layer_dropout = layer_dropout
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            if self.enable_dropout == True:
                self.dropout_layer = create_dropout_layer(self.dropout, self.num_gpus, self.default_gpu_id)
            
            if unit_dim % num_head != 0 or unit_dim / num_head == 0:
                raise ValueError("unit dim {0} and # head {1} mis-match".format(unit_dim, num_head))
            
            head_dim = unit_dim / num_head
            att_dim_list = []
            for i in range(num_head):
                att_dim = [head_dim, head_dim, head_dim]
                att_dim_list.append(att_dim)
            
            attention_layer_dropout = self.layer_dropout * float(self.sublayer_skip) / self.num_sublayer
            self.attention_layer = create_attention_layer("multi_head_att", self.unit_dim,
                self.unit_dim, att_dim_list, "scaled_dot", attention_layer_dropout, True, True, True,
                None, self.num_gpus, self.default_gpu_id, self.regularizer, self.random_seed, self.trainable)
            
            dense_layer_dropout = [self.layer_dropout * float(self.sublayer_skip + 1) / self.num_sublayer]
            self.dense_layer = create_dense_layer("double", 1, self.unit_dim, 4, self.activation, [self.dropout],
                dense_layer_dropout, True, True, num_gpus, default_gpu_id, self.regularizer, self.random_seed, self.trainable)
    
    def __call__(self,
                 input_src_data,
                 input_trg_data,
                 input_src_mask,
                 input_trg_mask):
        """call attentive-module layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            if self.enable_dropout == True:
                input_src_data, input_src_mask = self.dropout_layer(input_src_data, input_src_mask)
                input_trg_data, input_trg_mask = self.dropout_layer(input_trg_data, input_trg_mask)
            
            input_attention, input_attention_mask = self.attention_layer(input_src_data, input_trg_data, input_src_mask, input_trg_mask)
            input_dense, input_dense_mask = self.dense_layer(input_attention, input_attention_mask)
            
            output_module = input_dense
            output_module_mask = input_dense_mask
        
        return output_module, output_module_mask

class StackedAttentiveModule(object):
    """stacked attentive-module layer"""
    def __init__(self,
                 num_layer,
                 num_head,
                 unit_dim,
                 activation,
                 dropout,
                 layer_dropout,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="stacked_att_module"):
        """initialize stacked attentive-module layer"""
        self.num_layer = num_layer
        self.num_head = num_head
        self.unit_dim = unit_dim
        self.activation = activation
        self.dropout = dropout
        self.layer_dropout = layer_dropout
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.module_layer_list = []
            num_sublayer = 2 * self.num_layer
            for i in range(self.num_layer):
                layer_scope = "layer_{0}".format(i)
                enable_dropout = True if i % 2 == 0 else False
                sublayer_skip = 2 * i
                layer_default_gpu_id = self.default_gpu_id + i
                module_layer = AttentiveModule(num_head=self.num_head, unit_dim=self.unit_dim, activation=self.activation,
                    dropout=(enable_dropout, self.dropout), layer_dropout=(sublayer_skip, num_sublayer, self.layer_dropout),
                    num_gpus=self.num_gpus, default_gpu_id=layer_default_gpu_id, regularizer=self.regularizer,
                    random_seed=self.random_seed, trainable=self.trainable, scope=layer_scope)
                self.module_layer_list.append(module_layer)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call stacked attentive-module layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            output_module_list = []
            output_module_mask_list = []
            input_module = input_data
            input_module_mask = input_mask
            for module_layer in self.module_layer_list:
                output_module, output_module_mask = module_layer(input_module, input_module, input_module_mask, input_module_mask)
                input_module = output_module
                input_module_mask = output_module_mask
                output_module_list.append(output_module)
                output_module_mask_list.append(output_module_mask)
        
        return output_module_list, output_module_mask_list

class MultiAttentiveModule(object):
    """mutiple attentive-module layer"""
    def __init__(self,
                 num_layer,
                 num_head,
                 unit_dim,
                 activation,
                 dropout,
                 layer_dropout,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="mult_att_module"):
        """initialize mutiple attentive-module layer"""
        self.num_layer = num_layer
        self.num_head = num_head
        self.unit_dim = unit_dim
        self.activation = activation
        self.dropout = dropout
        self.layer_dropout = layer_dropout
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.module_layer_list = []
            for i in range(self.num_layer):
                layer_scope = "layer_{0}".format(i)
                layer_default_gpu_id = self.default_gpu_id + i
                module_layer = AttentiveModule(num_head=self.num_head, unit_dim=self.unit_dim,
                    activation=self.activation, dropout=(True, self.dropout), layer_dropout=(0, 2, self.layer_dropout),
                    num_gpus=self.num_gpus, default_gpu_id=layer_default_gpu_id, regularizer=self.regularizer,
                    random_seed=self.random_seed, trainable=self.trainable, scope=layer_scope)
                self.module_layer_list.append(module_layer)
    
    def __call__(self,
                 input_src_data,
                 input_trg_data,
                 input_src_mask,
                 input_trg_mask):
        """call multiple attentive-module layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            layer_size = len(self.module_layer_list)
            if (len(input_src_data) != layer_size or len(input_trg_data) != layer_size or
                len(input_src_mask) != layer_size or len(input_trg_mask) != layer_size):
                raise ValueError("input source & target list size and layer list size must be the same {0}".format(layer_size))
            
            output_module_list = []
            output_module_mask_list = []
            for i, module_layer in enumerate(self.module_layer_list):
                input_src_module = input_src_data[i]
                input_trg_module = input_trg_data[i]
                input_src_module_mask = input_src_mask[i]
                input_trg_module_mask = input_trg_mask[i]
                output_module, output_module_mask = module_layer(input_src_module,
                    input_trg_module, input_src_module_mask, input_trg_module_mask)
                output_module_list.append(output_module)
                output_module_mask_list.append(output_module_mask)
        
        return output_module_list, output_module_mask_list

class AggregationModule(object):
    """aggregation-module layer"""
    def __init__(self,
                 num_channel,
                 num_filter,
                 activation,
                 conv_window,
                 conv_stride,
                 pool_window,
                 pool_stride,
                 pooling_type,
                 dropout,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="agg_module"):
        """initialize aggregation-module layer"""
        self.num_channel = num_channel
        self.num_filter = num_filter
        self.activation = activation
        self.conv_window = conv_window
        self.conv_stride = conv_stride
        self.pool_window = pool_window
        self.pool_stride = pool_stride
        self.pooling_type = pooling_type
        self.dropout = dropout
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.dropout_layer = create_dropout_layer(self.dropout, self.num_gpus, self.default_gpu_id)
            
            self.conv_layer = create_convolution_layer("stacked_3d", 1, self.num_channel,
                self.num_filter, self.conv_window, self.conv_stride, "SAME", self.activation, [self.dropout], None,
                False, False, self.num_gpus, self.default_gpu_id, self.regularizer, self.random_seed, self.trainable)
            
            pooling_type = "{0}_3d".format(self.pooling_type)
            self.pooling_layer = create_pooling_layer(pooling_type, self.pool_window, self.pool_stride, 0, 0)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call aggregation-module layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_dropout, input_dropout_mask = self.dropout_layer(input_data, input_mask)
            
            input_conv, input_conv_mask = self.conv_layer(input_dropout, input_dropout_mask)
            input_pool, input_pool_mask = self.pooling_layer(input_conv, input_conv_mask)
            
            output_module = input_pool
            output_module_mask = input_pool_mask
        
        return output_module, output_module_mask

class StackedAggregationModule(object):
    """stacked aggregation-module layer"""
    def __init__(self,
                 num_layer,
                 unit_dim,
                 activation,
                 conv_window,
                 conv_stride,
                 pool_window,
                 pool_stride,
                 pooling_type,
                 dropout,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="stacked_agg_module"):
        """initialize stacked aggregation-module layer"""
        self.num_layer = num_layer
        self.num_channel = unit_dim[:-1]
        self.num_filter = unit_dim[1:]
        self.activation = activation
        self.conv_window = conv_window
        self.conv_stride = conv_stride
        self.pool_window = pool_window
        self.pool_stride = pool_stride
        self.pooling_type = pooling_type
        self.dropout = dropout
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.module_layer_list = []
            for i in range(self.num_layer):
                layer_scope = "layer_{0}".format(i)
                layer_default_gpu_id = self.default_gpu_id + i
                module_layer = AggregationModule(num_channel=self.num_channel[i], num_filter=self.num_filter[i],
                    activation=self.activation, conv_window=self.conv_window[i], conv_stride=self.conv_stride[i],
                    pool_window=self.pool_window[i], pool_stride=self.pool_stride[i], pooling_type=self.pooling_type[i],
                    dropout=self.dropout[i], num_gpus=self.num_gpus, default_gpu_id=layer_default_gpu_id,
                    regularizer=self.regularizer, random_seed=self.random_seed, trainable=self.trainable[i], scope=layer_scope)
                self.module_layer_list.append(module_layer)
    
    def __call__(self,
                 input_data,
                 input_mask):
        """call stacked aggregation-module layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_module = input_data
            input_module_mask = input_mask
            for module_layer in self.module_layer_list:
                input_module, input_module_mask = module_layer(input_module, input_module_mask)
            
            output_module = input_module
            output_module_mask = input_module_mask
        
        return output_module, output_module_mask

class WordFeat(object):
    """word-level featurization layer"""
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 dropout,
                 pretrained,
                 random_seed=0,
                 trainable=True,
                 scope="word_feat"):
        """initialize word-level featurization layer"""
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.pretrained = pretrained
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.embedding_layer = create_embedding_layer(self.vocab_size,
                self.embed_dim, self.pretrained, 0, 0, self.random_seed, self.trainable)
            
            self.dropout_layer = create_dropout_layer(self.dropout, 0, 0)
    
    def __call__(self,
                 input_word,
                 input_word_mask):
        """call word-level featurization layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_word_embedding_mask = input_word_mask
            input_word_embedding = tf.squeeze(self.embedding_layer(input_word), axis=-2)
            
            (input_word_dropout,
                input_word_dropout_mask) = self.dropout_layer(input_word_embedding, input_word_embedding_mask)
            
            input_word_feat = input_word_dropout
            input_word_feat_mask = input_word_dropout_mask
        
        return input_word_feat, input_word_feat_mask
    
    def get_embedding_placeholder(self):
        """get word-level embedding placeholder"""
        return self.embedding_layer.get_embedding_placeholder()

class CharFeat(object):
    """char-level featurization layer"""
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 unit_dim,
                 window_size,
                 activation,
                 pooling_type,
                 dropout,
                 num_gpus=1,
                 default_gpu_id=0,
                 regularizer=None,
                 random_seed=0,
                 trainable=True,
                 scope="char_feat"):
        """initialize char-level featurization layer"""
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.unit_dim = unit_dim
        self.window_size = window_size
        self.activation = activation
        self.pooling_type = pooling_type
        self.dropout = dropout
        self.num_gpus = num_gpus
        self.default_gpu_id = default_gpu_id
        self.regularizer = regularizer
        self.random_seed = random_seed
        self.trainable = trainable
        self.scope = scope
        
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.embedding_layer = create_embedding_layer(self.vocab_size,
                self.embed_dim, False, 0, 0, self.random_seed, self.trainable)
            
            self.conv_layer = create_convolution_layer("stacked_multi_1d", 1, self.embed_dim,
                self.unit_dim, self.window_size, 1, "SAME", self.activation, [self.dropout], None,
                False, False, self.num_gpus, self.default_gpu_id, self.regularizer, self.random_seed, self.trainable)
            
            self.pooling_layer = create_pooling_layer(self.pooling_type, -1, 1, 0, 0)
    
    def __call__(self,
                 input_char,
                 input_char_mask):
        """call char-level featurization layer"""
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            input_char_embedding_mask = tf.expand_dims(input_char_mask, axis=-1)
            input_char_embedding = self.embedding_layer(input_char)
            
            (input_char_conv,
                input_char_conv_mask) = self.conv_layer(input_char_embedding, input_char_embedding_mask)
            (input_char_pool,
                input_char_pool_mask) = self.pooling_layer(input_char_conv, input_char_conv_mask)
            
            input_char_feat = input_char_pool
            input_char_feat_mask = input_char_pool_mask
        
        return input_char_feat, input_char_feat_mask

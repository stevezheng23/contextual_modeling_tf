import codecs
import collections
import os.path
import json

import numpy as np
import tensorflow as tf

from util.default_util import *

__all__ = ["DataPipeline", "create_dynamic_pipeline", "create_data_pipeline",
           "create_src_dataset", "create_trg_dataset", "generate_word_feat", "generate_char_feat", "generate_num_feat",
           "create_embedding_file", "load_embedding_file", "convert_embedding",
           "create_vocab_file", "load_vocab_file", "process_vocab_table", "create_word_vocab", "create_char_vocab",
           "load_tsv_data", "load_json_data", "load_contextual_data", "prepare_data", "prepare_contextual_data"]

class DataPipeline(collections.namedtuple("DataPipeline",
    ("initializer", "input_context_word", "input_context_char", "input_response_word",
     "input_response_char", "input_label", "input_context_word_mask", "input_context_char_mask",
     "input_response_word_mask", "input_response_char_mask", "input_label_mask", "input_context_placeholder",
     "input_response_placeholder", "input_label_placeholder", "data_size_placeholder", "batch_size_placeholder"))):
    pass

def create_dynamic_pipeline(input_context_word_dataset,
                            input_context_char_dataset,
                            input_response_word_dataset,
                            input_response_char_dataset,
                            input_label_dataset,
                            word_vocab_index,
                            word_pad,
                            word_feat_enable,
                            char_vocab_index,
                            char_pad,
                            char_feat_enable,
                            input_context_placeholder,
                            input_response_placeholder,
                            input_label_placeholder,
                            data_size_placeholder,
                            batch_size_placeholder):
    """create dynamic data pipeline for contextual model"""
    default_pad_id = tf.constant(0, shape=[], dtype=tf.int32)
    default_dataset_tensor = tf.constant(0, shape=[1,1], dtype=tf.int32)
    
    if word_feat_enable == True:
        word_pad_id = tf.cast(word_vocab_index.lookup(tf.constant(word_pad)), dtype=tf.int32)
    else:
        word_pad_id = default_pad_id
        input_context_word_dataset = tf.data.Dataset.from_tensors(default_dataset_tensor).repeat(data_size_placeholder)
        input_response_word_dataset = tf.data.Dataset.from_tensors(default_dataset_tensor).repeat(data_size_placeholder)
    
    if char_feat_enable == True:
        char_pad_id = tf.cast(char_vocab_index.lookup(tf.constant(char_pad)), dtype=tf.int32)
    else:
        char_pad_id = default_pad_id
        input_context_char_dataset = tf.data.Dataset.from_tensors(default_dataset_tensor).repeat(data_size_placeholder)
        input_response_char_dataset = tf.data.Dataset.from_tensors(default_dataset_tensor).repeat(data_size_placeholder)
        
    dataset = tf.data.Dataset.zip((input_context_word_dataset, input_context_char_dataset,
        input_response_word_dataset, input_response_char_dataset, input_label_dataset))
    
    dataset = dataset.batch(batch_size=batch_size_placeholder)
    
    iterator = dataset.make_initializable_iterator()
    batch_data = iterator.get_next()
    
    if word_feat_enable == True:
        input_context_word = batch_data[0]
        input_context_word_mask = tf.cast(tf.not_equal(batch_data[0], word_pad_id), dtype=tf.float32)
        input_response_word = batch_data[2]
        input_response_word_mask = tf.cast(tf.not_equal(batch_data[2], word_pad_id), dtype=tf.float32)
    else:
        input_context_word = None
        input_context_word_mask = None
        input_response_word = None
        input_response_word_mask = None
    
    if char_feat_enable == True:
        input_context_char = batch_data[1]
        input_context_char_mask = tf.cast(tf.not_equal(batch_data[1], char_pad_id), dtype=tf.float32)
        input_response_char = batch_data[3]
        input_response_char_mask = tf.cast(tf.not_equal(batch_data[3], char_pad_id), dtype=tf.float32)
    else:
        input_context_char = None
        input_context_char_mask = None
        input_response_char = None
        input_response_char_mask = None
    
    label_pad_id = tf.constant(0, shape=[], dtype=tf.int32)
    input_label = tf.cast(batch_data[4], dtype=tf.float32)
    input_label_mask = tf.cast(tf.greater_equal(batch_data[4], label_pad_id), dtype=tf.float32)
    
    return DataPipeline(initializer=iterator.initializer,
        input_context_word=input_context_word, input_context_char=input_context_char,
        input_response_word=input_response_word, input_response_char=input_response_char, input_label=input_label,
        input_context_word_mask=input_context_word_mask, input_context_char_mask=input_context_char_mask,
        input_response_word_mask=input_response_word_mask, input_response_char_mask=input_response_char_mask,
        input_label_mask=input_label_mask, input_context_placeholder=input_context_placeholder,
        input_response_placeholder=input_response_placeholder, input_label_placeholder=input_label_placeholder,
        data_size_placeholder=data_size_placeholder, batch_size_placeholder=batch_size_placeholder)

def create_data_pipeline(input_context_word_dataset,
                         input_context_char_dataset,
                         input_response_word_dataset,
                         input_response_char_dataset,
                         input_label_dataset,
                         word_vocab_index,
                         word_pad,
                         word_feat_enable,
                         char_vocab_index,
                         char_pad,
                         char_feat_enable,
                         enable_shuffle,
                         buffer_size,
                         data_size,
                         batch_size,
                         random_seed):
    """create data pipeline for contextual model"""
    default_pad_id = tf.constant(0, shape=[], dtype=tf.int32)
    default_dataset_tensor = tf.constant(0, shape=[1,1], dtype=tf.int32)
    
    if word_feat_enable == True:
        word_pad_id = tf.cast(word_vocab_index.lookup(tf.constant(word_pad)), dtype=tf.int32)
    else:
        word_pad_id = default_pad_id
        input_context_word_dataset = tf.data.Dataset.from_tensors(default_dataset_tensor).repeat(data_size)
        input_response_word_dataset = tf.data.Dataset.from_tensors(default_dataset_tensor).repeat(data_size)
    
    if char_feat_enable == True:
        char_pad_id = tf.cast(char_vocab_index.lookup(tf.constant(char_pad)), dtype=tf.int32)
    else:
        char_pad_id = default_pad_id
        input_context_char_dataset = tf.data.Dataset.from_tensors(default_dataset_tensor).repeat(data_size)
        input_response_char_dataset = tf.data.Dataset.from_tensors(default_dataset_tensor).repeat(data_size)
    
    dataset = tf.data.Dataset.zip((input_context_word_dataset, input_context_char_dataset,
        input_response_word_dataset, input_response_char_dataset, input_label_dataset))
    
    if enable_shuffle == True:
        buffer_size = min(buffer_size, data_size)
        dataset = dataset.shuffle(buffer_size, random_seed)
    
    dataset = dataset.batch(batch_size=batch_size)
    
    iterator = dataset.make_initializable_iterator()
    batch_data = iterator.get_next()
    
    if word_feat_enable == True:
        input_context_word = batch_data[0]
        input_context_word_mask = tf.cast(tf.not_equal(batch_data[0], word_pad_id), dtype=tf.float32)
        input_response_word = batch_data[2]
        input_response_word_mask = tf.cast(tf.not_equal(batch_data[2], word_pad_id), dtype=tf.float32)
    else:
        input_context_word = None
        input_context_word_mask = None
        input_response_word = None
        input_response_word_mask = None
    
    if char_feat_enable == True:
        input_context_char = batch_data[1]
        input_context_char_mask = tf.cast(tf.not_equal(batch_data[1], char_pad_id), dtype=tf.float32)
        input_response_char = batch_data[3]
        input_response_char_mask = tf.cast(tf.not_equal(batch_data[3], char_pad_id), dtype=tf.float32)
    else:
        input_context_char = None
        input_context_char_mask = None
        input_response_char = None
        input_response_char_mask = None
    
    label_pad_id = tf.constant(0, shape=[], dtype=tf.int32)
    input_label = tf.cast(batch_data[4], dtype=tf.float32)
    input_label_mask = tf.cast(tf.greater_equal(batch_data[4], label_pad_id), dtype=tf.float32)
    
    return DataPipeline(initializer=iterator.initializer,
        input_context_word=input_context_word, input_context_char=input_context_char,
        input_response_word=input_response_word, input_response_char=input_response_char, input_label=input_label,
        input_context_word_mask=input_context_word_mask, input_context_char_mask=input_context_char_mask,
        input_response_word_mask=input_response_word_mask, input_response_char_mask=input_response_char_mask,
        input_label_mask=input_label_mask, input_context_placeholder=None, input_response_placeholder=None,
        input_label_placeholder=None, data_size_placeholder=None, batch_size_placeholder=None)
    
def create_src_dataset(input_data_set,
                       sentence_max_backward,
                       sentence_max_size,
                       word_vocab_index,
                       word_max_size,
                       word_pad,
                       word_feat_enable,
                       char_vocab_index,
                       char_max_size,
                       char_pad,
                       char_feat_enable):
    """create word/char-level dataset for input source data"""
    dataset = input_data_set
    
    word_dataset = None
    if word_feat_enable == True:
        word_dataset = dataset.map(lambda para: generate_word_feat(para, sentence_max_backward, sentence_max_size,
            word_vocab_index, word_max_size, word_pad))
    
    char_dataset = None
    if char_feat_enable == True:
        char_dataset = dataset.map(lambda para: generate_char_feat(para, sentence_max_backward, sentence_max_size,
            word_max_size, char_vocab_index, char_max_size, char_pad))
    
    return word_dataset, char_dataset

def create_trg_dataset(input_data_set,
                       string_max_backward,
                       string_max_size):
    """create label dataset for input target data"""
    dataset = input_data_set
    
    num_dataset = dataset.map(lambda para: generate_num_feat(para, string_max_backward, string_max_size))
    
    return num_dataset

def generate_word_feat(paragraph,
                       sentence_max_backward,
                       sentence_max_size,
                       word_vocab_index,
                       word_max_size,
                       word_pad):
    """process words for paragraph"""
    def sent_to_word(sentence):
        """process words for sentence"""
        words = tf.string_split([sentence], delimiter=' ').values
        words = tf.concat([words[:word_max_size],
            tf.constant(word_pad, shape=[word_max_size])], axis=0)
        words = tf.reshape(words[:word_max_size], shape=[word_max_size])
        
        return words
    
    sentences = tf.string_split([paragraph], delimiter='|').values
    sentences = sentences[-sentence_max_size:] if sentence_max_backward else sentences[:sentence_max_size]
    sentences = tf.concat([sentences, tf.constant(word_pad, shape=[sentence_max_size])], axis=0)
    sentences = tf.reshape(sentences[:sentence_max_size], shape=[sentence_max_size])
    sentence_words = tf.map_fn(sent_to_word, sentences)
    sentence_words = tf.cast(word_vocab_index.lookup(sentence_words), dtype=tf.int32)
    sentence_words = tf.expand_dims(sentence_words, axis=-1)
    
    return sentence_words

def generate_char_feat(paragraph,
                       sentence_max_backward,
                       sentence_max_size,
                       word_max_size,
                       char_vocab_index,
                       char_max_size,
                       char_pad):
    """generate characters for paragraph"""
    def sent_to_char(sentence):
        """process chars for sentence"""
        words = tf.string_split([sentence], delimiter=' ').values
        words = tf.concat([words[:word_max_size],
            tf.constant(char_pad, shape=[word_max_size])], axis=0)
        words = tf.reshape(words[:word_max_size], shape=[word_max_size])
        word_chars = tf.map_fn(word_to_char, words)
        
        return word_chars
    
    def word_to_char(word):
        """process characters for word"""
        chars = tf.string_split([word], delimiter='').values
        chars = tf.concat([chars[:char_max_size],
            tf.constant(char_pad, shape=[char_max_size])], axis=0)
        chars = tf.reshape(chars[:char_max_size], shape=[char_max_size])
        
        return chars
    
    sentences = tf.string_split([paragraph], delimiter='|').values
    sentences = sentences[-sentence_max_size:] if sentence_max_backward else sentences[:sentence_max_size]
    sentences = tf.concat([sentences, tf.constant(char_pad, shape=[sentence_max_size])], axis=0)
    sentences = tf.reshape(sentences[:sentence_max_size], shape=[sentence_max_size])
    sentence_chars = tf.map_fn(sent_to_char, sentences)
    sentence_chars = tf.cast(char_vocab_index.lookup(sentence_chars), dtype=tf.int32)
    
    return sentence_chars

def generate_num_feat(paragraph,
                      string_max_backward,
                      string_max_size):
    """generate numbers for paragraph"""
    strings = tf.string_split([paragraph], delimiter='|').values
    strings = strings[-string_max_size:] if string_max_backward else strings[:string_max_size]
    strings = tf.concat([strings, tf.constant("0", shape=[string_max_size])], axis=0)
    strings = tf.reshape(strings[:string_max_size], shape=[string_max_size])
    string_nums = tf.string_to_number(strings, out_type=tf.int32)
    string_nums = tf.expand_dims(string_nums, axis=-1)
    
    return string_nums

def create_embedding_file(embedding_file,
                          embedding_table):
    """create embedding file based on embedding table"""
    embedding_dir = os.path.dirname(embedding_file)
    if not tf.gfile.Exists(embedding_dir):
        tf.gfile.MakeDirs(embedding_dir)
    
    if not tf.gfile.Exists(embedding_file):
        with codecs.getwriter("utf-8")(open(embedding_file, "wb")) as file:
            for vocab in embedding_table.keys():
                embed = embedding_table[vocab]
                embed_str = " ".join(map(str, embed))
                file.write("{0} {1}\n".format(vocab, embed_str))

def load_embedding_file(embedding_file,
                        embedding_size,
                        unk,
                        pad):
    """load pre-train embeddings from embedding file"""
    if tf.gfile.Exists(embedding_file):
        with codecs.getreader("utf-8")(open(embedding_file, "rb")) as file:
            embedding = {}
            for line in file:
                items = line.strip().split(' ')
                if len(items) != embedding_size + 1:
                    continue
                word = items[0]
                vector = [float(x) for x in items[1:]]
                if word not in embedding:
                    embedding[word] = vector
            
            if unk not in embedding:
                embedding[unk] = np.random.rand(embedding_size)
            if pad not in embedding:
                embedding[pad] = np.random.rand(embedding_size)
            
            return embedding
    else:
        raise FileNotFoundError("embedding file not found")

def convert_embedding(embedding_lookup):
    if embedding_lookup is not None:
        embedding = [v for k,v in embedding_lookup.items()]
    else:
        embedding = None
    
    return embedding

def create_vocab_file(vocab_file,
                      vocab_table):
    """create vocab file based on vocab table"""
    vocab_dir = os.path.dirname(vocab_file)
    if not tf.gfile.Exists(vocab_dir):
        tf.gfile.MakeDirs(vocab_dir)
    
    if not tf.gfile.Exists(vocab_file):
        with codecs.getwriter("utf-8")(open(vocab_file, "wb")) as file:
            for vocab in vocab_table:
                file.write("{0}\n".format(vocab))

def load_vocab_file(vocab_file):
    """load vocab data from vocab file"""
    if tf.gfile.Exists(vocab_file):
        with codecs.getreader("utf-8")(open(vocab_file, "rb")) as file:
            vocab = {}
            for line in file:
                items = line.strip().split('\t')
                
                item_size = len(items)
                if item_size > 1:
                    vocab[items[0]] = int(items[1])
                elif item_size > 0:
                    vocab[items[0]] = MAX_INT
            
            return vocab
    else:
        raise FileNotFoundError("vocab file not found")

def process_vocab_table(vocab,
                        vocab_size,
                        vocab_threshold,
                        vocab_lookup,
                        unk,
                        pad):
    """process vocab table"""
    default_vocab = [unk, pad]
    
    if unk in vocab:
        del vocab[unk]
    if pad in vocab:
        del vocab[pad]
    
    vocab = { k: vocab[k] for k in vocab.keys() if vocab[k] >= vocab_threshold }
    if vocab_lookup is not None:
        vocab = { k: vocab[k] for k in vocab.keys() if k in vocab_lookup }
    
    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    sorted_vocab = default_vocab + sorted_vocab
    
    vocab_table = sorted_vocab[:vocab_size]
    vocab_size = len(vocab_table)
    
    vocab_index = tf.contrib.lookup.index_table_from_tensor(
        mapping=tf.constant(vocab_table), default_value=0)
    vocab_inverted_index = tf.contrib.lookup.index_to_string_table_from_tensor(
        mapping=tf.constant(vocab_table), default_value=unk)
    
    return vocab_table, vocab_size, vocab_index, vocab_inverted_index

def create_word_vocab(input_data):
    """create word vocab from input data"""
    sentence_separator = "|"
    word_vocab = {}
    for paragraph in input_data:
        sentences = paragraph.strip().split(sentence_separator)
        for sentence in sentences:
            words = sentence.strip().split(' ')
            for word in words:
                if word not in word_vocab:
                    word_vocab[word] = 1
                else:
                    word_vocab[word] += 1
    
    return word_vocab

def create_char_vocab(input_data):
    """create char vocab from input data"""
    sentence_separator = "|"
    char_vocab = {}
    for paragraph in input_data:
        sentences = paragraph.strip().split(sentence_separator)
        for sentence in sentences:
            words = sentence.strip().split(' ')
            for word in words:
                chars = list(word)
                for ch in chars:
                    if ch not in char_vocab:
                        char_vocab[ch] = 1
                    else:
                        char_vocab[ch] += 1
    
    return char_vocab

def load_tsv_data(input_file):
    """load data from tsv file"""
    if tf.gfile.Exists(input_file):
        context_data = []
        response_data = []
        label_data = []
        context_lookup = {}
        item_separator = "\t"
        subitem_separator = "|"
        with codecs.getreader("utf-8")(open(input_file, "rb")) as file:
            for line in file:
                items = line.strip().split(item_separator)
                if len(items) < 4:
                    continue
                
                sample_id = items[0]
                context = items[1]
                if context not in context_lookup:
                    context_lookup[context] = {
                        "id": sample_id,
                        "context": context,
                        "response": []
                    }
                
                response = {
                    "text": items[2],
                    "label": items[3]
                }
                
                context_lookup[context]["response"].append(response)
            
            for context in context_lookup.keys():
                response = subitem_separator.join(context_lookup[context]["response"])
                label = subitem_separator.join(context_lookup[context]["label"])
                context_data.append(context)
                response_data.append(response)
                label_data.append(label)
            
            input_data = context_lookup.values()
        
        return input_data, context_data, response_data, label_data
    else:
        raise FileNotFoundError("input file not found")

def load_json_data(input_file):
    """load data from json file"""
    if tf.gfile.Exists(input_file):
        context_data = []
        response_data = []
        label_data = []
        subitem_separator = "|"
        with codecs.getreader("utf-8")(open(input_file, "rb") ) as file:
            input_data = json.load(file)
            for item in input_data:
                context = subitem_separator.join(item["context"])
                response = subitem_separator.join([r["text"] for r in item["response"]])
                label = subitem_separator.join([r["label"] for r in item["response"]])
                context_data.append(context)
                response_data.append(response)
                label_data.append(label)
        
        return input_data, context_data, response_data, label_data
    else:
        raise FileNotFoundError("input file not found")

def load_contextual_data(input_file,
                         file_type):
    """load contextual data from input file"""
    if file_type == "tsv":
        input_data, context_data, response_data, label_data = load_tsv_data(input_file)
    elif file_type == "json":
        input_data, context_data, response_data, label_data = load_json_data(input_file)
    else:
        raise ValueError("can not load data from unsupported file type {0}".format(file_type))
    
    return input_data, context_data, response_data, label_data

def prepare_data(logger,
                 input_data,
                 word_vocab_file,
                 word_vocab_size,
                 word_vocab_threshold,
                 word_embed_dim,
                 word_embed_file,
                 full_word_embed_file,
                 word_unk,
                 word_pad,
                 word_feat_enable,
                 pretrain_word_embed,
                 char_vocab_file,
                 char_vocab_size,
                 char_vocab_threshold,
                 char_unk,
                 char_pad,
                 char_feat_enable):
    """prepare data"""    
    word_embed_data = None
    if pretrain_word_embed == True:
        if tf.gfile.Exists(word_embed_file):
            logger.log_print("# loading word embeddings from {0}".format(word_embed_file))
            word_embed_data = load_embedding_file(word_embed_file, word_embed_dim, word_unk, word_pad)
        elif tf.gfile.Exists(full_word_embed_file):
            logger.log_print("# loading word embeddings from {0}".format(full_word_embed_file))
            word_embed_data = load_embedding_file(full_word_embed_file, word_embed_dim, word_unk, word_pad)
        else:
            raise ValueError("{0} or {1} must be provided".format(word_vocab_file, full_word_embed_file))
        
        word_embed_size = len(word_embed_data) if word_embed_data is not None else 0
        logger.log_print("# word embedding table has {0} words".format(word_embed_size))
    
    word_vocab = None
    word_vocab_index = None
    word_vocab_inverted_index = None
    if tf.gfile.Exists(word_vocab_file):
        logger.log_print("# loading word vocab table from {0}".format(word_vocab_file))
        word_vocab = load_vocab_file(word_vocab_file)
        (word_vocab_table, word_vocab_size, word_vocab_index,
            word_vocab_inverted_index) = process_vocab_table(word_vocab, word_vocab_size,
            word_vocab_threshold, word_embed_data, word_unk, word_pad)
    elif input_data is not None:
        logger.log_print("# creating word vocab table from input data")
        word_vocab = create_word_vocab(input_data)
        (word_vocab_table, word_vocab_size, word_vocab_index,
            word_vocab_inverted_index) = process_vocab_table(word_vocab, word_vocab_size,
            word_vocab_threshold, word_embed_data, word_unk, word_pad)
        logger.log_print("# creating word vocab file {0}".format(word_vocab_file))
        create_vocab_file(word_vocab_file, word_vocab_table)
    else:
        raise ValueError("{0} or input data must be provided".format(word_vocab_file))
    
    logger.log_print("# word vocab table has {0} words".format(word_vocab_size))
    
    char_vocab = None
    char_vocab_index = None
    char_vocab_inverted_index = None
    if char_feat_enable is True:
        if tf.gfile.Exists(char_vocab_file):
            logger.log_print("# loading char vocab table from {0}".format(char_vocab_file))
            char_vocab = load_vocab_file(char_vocab_file)
            (_, char_vocab_size, char_vocab_index,
                char_vocab_inverted_index) = process_vocab_table(char_vocab, char_vocab_size,
                char_vocab_threshold, None, char_unk, char_pad)
        elif input_data is not None:
            logger.log_print("# creating char vocab table from input data")
            char_vocab = create_char_vocab(input_data)
            (char_vocab_table, char_vocab_size, char_vocab_index,
                char_vocab_inverted_index) = process_vocab_table(char_vocab, char_vocab_size,
                char_vocab_threshold, None, char_unk, char_pad)
            logger.log_print("# creating char vocab file {0}".format(char_vocab_file))
            create_vocab_file(char_vocab_file, char_vocab_table)
        else:
            raise ValueError("{0} or input data must be provided".format(char_vocab_file))

        logger.log_print("# char vocab table has {0} chars".format(char_vocab_size))
    
    if word_embed_data is not None and word_vocab_table is not None:
        word_embed_data = { k: word_embed_data[k] for k in word_vocab_table if k in word_embed_data }
        logger.log_print("# word embedding table has {0} words after filtering".format(len(word_embed_data)))
        if not tf.gfile.Exists(word_embed_file):
            logger.log_print("# creating word embedding file {0}".format(word_embed_file))
            create_embedding_file(word_embed_file, word_embed_data)
        
        word_embed_data = convert_embedding(word_embed_data)
    
    return (word_embed_data, word_vocab_size, word_vocab_index, word_vocab_inverted_index,
        char_vocab_size, char_vocab_index, char_vocab_inverted_index)

def prepare_contextual_data(logger,
                            input_contextual_file,
                            input_file_type,
                            word_vocab_file,
                            word_vocab_size,
                            word_vocab_threshold,
                            word_embed_dim,
                            word_embed_file,
                            full_word_embed_file,
                            word_unk,
                            word_pad,
                            word_feat_enable,
                            pretrain_word_embed,
                            char_vocab_file,
                            char_vocab_size,
                            char_vocab_threshold,
                            char_unk,
                            char_pad,
                            char_feat_enable):
    """prepare contextual data"""
    input_data = set()
    logger.log_print("# loading input contextual data from {0}".format(input_contextual_file))
    (input_contextual_data, input_context_data, input_response_data,
        input_label_data) = load_contextual_data(input_contextual_file, input_file_type)
    
    input_contextual_size = len(input_contextual_data)
    input_context_size = len(input_context_data)
    input_response_size = len(input_response_data)
    input_label_size = len(input_label_data)
    logger.log_print("# input contextual data has {0} lines".format(input_contextual_size))
    
    if (input_context_size != input_contextual_size or input_response_size != input_contextual_size or
        input_label_size != input_contextual_size):
        raise ValueError("context, response & label input data must have the same size")
    
    input_data.update(input_context_data)
    input_data.update(input_response_data)
    
    input_data = list(input_data)
    (word_embed_data, word_vocab_size, word_vocab_index, word_vocab_inverted_index,
        char_vocab_size, char_vocab_index, char_vocab_inverted_index) = prepare_data(logger, input_data,
            word_vocab_file, word_vocab_size, word_vocab_threshold, word_embed_dim, word_embed_file,
            full_word_embed_file, word_unk, word_pad, word_feat_enable, pretrain_word_embed,
            char_vocab_file, char_vocab_size, char_vocab_threshold, char_unk, char_pad, char_feat_enable)
    
    return (input_contextual_data, input_context_data, input_response_data, input_label_data, word_embed_data,
        word_vocab_size, word_vocab_index, word_vocab_inverted_index, char_vocab_size, char_vocab_index, char_vocab_inverted_index)

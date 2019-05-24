import argparse
import os.path
import time

import numpy as np
import tensorflow as tf

from tensorflow.python import debug as tf_debug

from util.default_util import *
from util.param_util import *
from util.model_util import *
from util.eval_util import *
from util.debug_logger import *
from util.train_logger import *
from util.eval_logger import *
from util.summary_writer import *

def add_arguments(parser):
    parser.add_argument("--mode", help="mode to run", required=True)
    parser.add_argument("--config", help="path to json config", required=True)

def pipeline_initialize(sess,
                        model,
                        pipeline_mode,
                        batch_size):
    data_size = len(model.input_data)
    data_dict = {
        "data_size": data_size,
        "input_data": model.input_data,
        "input_context": model.input_context,
        "input_response": model.input_response,
        "input_label": model.input_label
    }
    
    if pipeline_mode == "dynamic":
        sess.run(model.data_pipeline.initializer,
            feed_dict={model.data_pipeline.input_context_placeholder: model.input_context,
                model.data_pipeline.input_response_placeholder: model.input_response,
                model.data_pipeline.input_label_placeholder: model.input_label,
                model.data_pipeline.data_size_placeholder: data_size,
                model.data_pipeline.batch_size_placeholder: batch_size})
    else:
        sess.run(model.data_pipeline.initializer)
    
    return data_dict

def extrinsic_eval(logger,
                   summary_writer,
                   sess,
                   model,
                   pipeline_mode,
                   batch_size,
                   metric_list,
                   global_step,
                   epoch,
                   ckpt_file,
                   eval_mode):
    load_model(sess, model, ckpt_file, eval_mode)
    data_dict = pipeline_initialize(sess, model, pipeline_mode, batch_size)
    
    data_size = data_dict["data_size"]
    input_data = data_dict["input_data"]
    sample_predict = []
    while True:
        try:
            infer_result = model.model.infer(sess)
            sample_predict.extend(infer_result.predict)
        except  tf.errors.OutOfRangeError:
            break
    
    sample_output = []
    predict_output = []
    label_output = []
    for i in range(data_size):
        sample = {
            "id": input_data[i]["id"],
            "context": input_data[i]["context"],
            "response": []
        }
        
        predict_list = []
        label_list = []
        for j, resp in enumerate(input_data[i]["response"]):
            predict = float(sample_predict[i][j,0])
            label = float(resp["label"])
            response = {
                "text": resp["text"],
                "predict": str(predict),
                "label": str(label)
            }
            
            sample["response"].append(response)
            predict_list.append(predict)
            label_list.append(label)
        
        sample_output.append(sample)
        predict_output.append(predict_list)
        label_output.append(label_list)
    
    eval_result_list = []
    for metric in metric_list:
        score = evaluate_from_data(predict_output, label_output, metric)
        summary_writer.add_value_summary(metric, score, global_step)
        eval_result = ExtrinsicEvalLog(metric=metric,
            score=score, sample_output=None, sample_size=len(sample_output))
        eval_result_list.append(eval_result)
    
    eval_result_detail = ExtrinsicEvalLog(metric="detail",
        score=0.0, sample_output=sample_output, sample_size=len(sample_output))
    basic_info = BasicInfoEvalLog(epoch=epoch, global_step=global_step)
    
    logger.update_extrinsic_eval(eval_result_list, basic_info)
    logger.update_extrinsic_eval_detail(eval_result_detail, basic_info)
    logger.check_extrinsic_eval()
    logger.check_extrinsic_eval_detail()

def train(logger,
          hyperparams,
          enable_eval=True,
          enable_debug=False):
    config_proto = get_config_proto(hyperparams.device_log_device_placement,
        hyperparams.device_allow_soft_placement, hyperparams.device_allow_growth,
        hyperparams.device_per_process_gpu_memory_fraction)
    
    summary_output_dir = hyperparams.train_summary_output_dir
    if not tf.gfile.Exists(summary_output_dir):
        tf.gfile.MakeDirs(summary_output_dir)
    
    logger.log_print("##### create train model #####")
    train_model = create_train_model(logger, hyperparams)
    train_sess = tf.Session(config=config_proto, graph=train_model.graph)
    if enable_debug == True:
        train_sess = tf_debug.LocalCLIDebugWrapperSession(train_sess)
    
    train_summary_writer = SummaryWriter(train_model.graph, os.path.join(summary_output_dir, "train"))
    init_model(train_sess, train_model)
    train_logger = TrainLogger(hyperparams.data_log_output_dir)
    
    if enable_eval == True:
        logger.log_print("##### create infer model #####")
        infer_model = create_infer_model(logger, hyperparams)
        infer_sess = tf.Session(config=config_proto, graph=infer_model.graph)
        if enable_debug == True:
            infer_sess = tf_debug.LocalCLIDebugWrapperSession(infer_sess)
        
        infer_summary_writer = SummaryWriter(infer_model.graph, os.path.join(summary_output_dir, "infer"))
        init_model(infer_sess, infer_model)
        eval_logger = EvalLogger(hyperparams.data_log_output_dir)
    
    logger.log_print("##### start training #####")
    global_step = 0
    for epoch in range(hyperparams.train_num_epoch):
        data_dict = pipeline_initialize(train_sess, train_model,
            hyperparams.data_pipeline_mode, hyperparams.train_batch_size)
        
        step_in_epoch = 0
        while True:
            try:
                start_time = time.time()
                train_result = train_model.model.train(train_sess)
                end_time = time.time()
                
                global_step = train_result.global_step
                step_in_epoch += 1
                train_logger.update(train_result, epoch, step_in_epoch, end_time-start_time)
                
                if step_in_epoch % hyperparams.train_step_per_stat == 0:
                    train_logger.check()
                    train_summary_writer.add_summary(train_result.summary, global_step)
                if step_in_epoch % hyperparams.train_step_per_ckpt == 0:
                    train_model.model.save(train_sess, global_step, "debug")
                if step_in_epoch % hyperparams.train_step_per_eval == 0 and enable_eval == True:
                    ckpt_file = infer_model.model.get_latest_ckpt("debug")
                    extrinsic_eval(eval_logger, infer_summary_writer, infer_sess, infer_model,
                        hyperparams.data_pipeline_mode, hyperparams.train_eval_batch_size, 
                        hyperparams.train_eval_metric, global_step, epoch, ckpt_file, "debug")
            except tf.errors.OutOfRangeError:
                train_logger.check()
                train_summary_writer.add_summary(train_result.summary, global_step)
                train_model.model.save(train_sess, global_step, "epoch")
                if enable_eval == True:
                    ckpt_file = infer_model.model.get_latest_ckpt("epoch")
                    extrinsic_eval(eval_logger, infer_summary_writer, infer_sess, infer_model,
                        hyperparams.data_pipeline_mode, hyperparams.train_eval_batch_size, 
                        hyperparams.train_eval_metric, global_step, epoch, ckpt_file, "epoch")
                break

    train_summary_writer.close_writer()
    if enable_eval == True:
        infer_summary_writer.close_writer()
    
    logger.log_print("##### finish training #####")

def evaluate(logger,
             hyperparams,
             enable_debug=False):   
    config_proto = get_config_proto(hyperparams.device_log_device_placement,
        hyperparams.device_allow_soft_placement, hyperparams.device_allow_growth,
        hyperparams.device_per_process_gpu_memory_fraction)
    
    summary_output_dir = hyperparams.train_summary_output_dir
    if not tf.gfile.Exists(summary_output_dir):
        tf.gfile.MakeDirs(summary_output_dir)
    
    logger.log_print("##### create infer model #####")
    infer_model = create_infer_model(logger, hyperparams)
    infer_sess = tf.Session(config=config_proto, graph=infer_model.graph)
    if enable_debug == True:
        infer_sess = tf_debug.LocalCLIDebugWrapperSession(infer_sess)
    
    infer_summary_writer = SummaryWriter(infer_model.graph, os.path.join(summary_output_dir, "infer"))
    init_model(infer_sess, infer_model)
    eval_logger = EvalLogger(hyperparams.data_log_output_dir)
    
    logger.log_print("##### start evaluation #####")
    global_step = 0
    eval_mode = "debug" if enable_debug == True else "epoch"
    ckpt_file_list = infer_model.model.get_ckpt_list(eval_mode)
    for i, ckpt_file in enumerate(ckpt_file_list):
        extrinsic_eval(eval_logger, infer_summary_writer, infer_sess, infer_model,
            hyperparams.data_pipeline_mode, hyperparams.train_eval_batch_size, 
            hyperparams.train_eval_metric, global_step, i, ckpt_file, eval_mode)
    
    infer_summary_writer.close_writer()
    logger.log_print("##### finish evaluation #####")

def main(args):
    hyperparams = load_hyperparams(args.config)
    logger = DebugLogger(hyperparams.data_log_output_dir)
    
    tf_version = check_tensorflow_version()
    logger.log_print("# tensorflow verison is {0}".format(tf_version))
    
    if (args.mode == 'train'):
        train(logger, hyperparams, enable_eval=False, enable_debug=False)
    elif (args.mode == 'train_eval'):
        train(logger, hyperparams, enable_eval=True, enable_debug=False)
    elif (args.mode == 'train_debug'):
        train(logger, hyperparams, enable_eval=False, enable_debug=True)
    elif (args.mode == 'eval'):
        evaluate(logger, hyperparams, enable_debug=False)
    elif (args.mode == 'eval_debug'):
        evaluate(logger, hyperparams, enable_debug=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    main(args)

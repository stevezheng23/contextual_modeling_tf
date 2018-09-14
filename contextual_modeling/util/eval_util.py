import numpy as np
import tensorflow as tf

from external.coverage_precision import *
from external.precision import *

__all__ = ["evaluate_from_data"]

def _cp_auc_at_k(pred_data, label_data, k):
    """Coverage-Precision AUC at top K"""
    eval_option = {
        "position": [k],
        "threshold": {
            "start": 0.0,
            "end": 1.0,
            "step": 0.001
        }
    }
    
    cp_auc = get_cp_auc(pred_data, label_data, eval_option)
    return cp_auc[k]

def _precision_at_k(pred_data, label_data, k):
    """Precision at K"""
    eval_option = { "position": [k] }
    precision = get_precision(pred_data, label_data, eval_option)
    return precision[k]

def evaluate_from_data(pred_data, label_data, metric):
    """compute evaluation score based on selected metric"""
    if len(pred_data) == 0 or len(label_data) == 0:
        return 0.0
    
    if metric == "cp_auc@1":
        eval_score = _cp_auc_at_k(pred_data, label_data, 1)
    elif metric == "cp_auc@3":
        eval_score = _cp_auc_at_k(pred_data, label_data, 3)
    elif metric == "cp_auc@5":
        eval_score = _cp_auc_at_k(pred_data, label_data, 5)
    elif metric == "precision@1":
        eval_score = _precision_at_k(pred_data, label_data, 1)
    elif metric == "precision@3":
        eval_score = _precision_at_k(pred_data, label_data, 3)
    elif metric == "precision@5":
        eval_score = _precision_at_k(pred_data, label_data, 5)
    else:
        raise ValueError("unsupported metric {0}".format(metric))
    
    return eval_score

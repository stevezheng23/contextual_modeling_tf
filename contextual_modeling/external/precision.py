import numpy as np

__all__ = ["get_precision"]

def get_precision(predict_data,
                  label_data,
                  eval_option):
    if len(predict_data) != len(label_data):
        raise ValueError('# predict and # label must be the same')
    
    sample_data = list(zip(predict_data, label_data))
    if len(sample_data) == 0:
        raise ValueError('# sample must be more 0')
    
    sample_candidates = []
    for predict_list, label_list in sample_data:
        if len(predict_list) != len(label_list):
            continue
        
        candidate_list = list(zip(predict_list, label_list))
        sample_candidates.append(candidate_list)
    
    position_list = eval_option["position"]
    position_list = sorted(position_list)
    
    covered = 0
    position_hit = { position: 0 for position in position_list }
    for candidates in sample_candidates:
        candidates = sorted(candidates, key=lambda candidate: candidate[0], reverse=True)
        covered += any(candidates)
        for position in position_list:
            position_hit[position] += any([candidate[1] > 0.0 for candidate in candidates[:position]])

    if covered > 0:
        precision = { position: float(position_hit[position]) / float(covered) for position in position_hit }
    else:
        precision = { position: 0.0 for position in position_hit }
    
    return precision

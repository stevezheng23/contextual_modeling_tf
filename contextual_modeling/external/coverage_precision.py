import numpy as np

__all__ = ["get_cp_auc"]

def get_cp_auc(predict_data,
               label_data,
               eval_option):
    if len(predict_data) != len(label_data):
        raise ValueError('# predict and # label must be the same')
    
    sample_data = zip(predict_data, label_data)
    if len(sample_data) == 0:
        raise ValueError('# sample must be more 0')
    
    sample_candidates = []
    for predicts, label_list in sample_data:
        if len(predict_list) != len(label_list):
            continue
        
        candidate_list = zip(predict_list, label_list)
        sample_candidates.append(candidate_list)
    
    position_list = eval_option["positions"]
    sorted(position_list)
    
    threshold_start = eval_option["threshold"]["start"]
    threshold_end = eval_option["threshold"]["end"]
    threshold_step = eval_option["threshold"]["step"]
    
    coverage_precision = {}
    total = len(sample_candidates)
    for threshold in np.arange(threshold_start, threshold_end, threshold_step):
        covered = 0
        position_hit = { position: 0 for position in position_list }
        for candidates in sample_candidates:
            candidates = [candidate for candidate in candidates if candidate[0] >= threshold]
            sorted(candidates, key=lambda candidate: candidate[0])
            covered += any(candidates)
            for position in position_list:
                position_hit[position] += any([candidate[1] == 1.0 for candidate in candidates[:position]])
        
        if covered > 0:
            precision = { position: float(position_hit[position]) / float(covered) for position in position_hit }
        else:
            precision = { position: 0.0 for position in position_hit }
        
        coverage = float(covered) / float(total)
        coverage_precision[coverage] = precision
    
    coverage_step = 0.1 * threshold_step
    coverage_list = coverage_precision.keys()
    sorted(coverage_list)
    prev_coverage = 0.0
    area_under_curve = 0.0
    for coverage in coverage_list:
        area_under_curve += (coverage - prev_coverage) * coverage_precision[coverage]
        prev_coverage = coverage
    
    return area_under_curve

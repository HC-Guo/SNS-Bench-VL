from pathlib import Path
import json
import tqdm
import yaml
import random
import numpy as np
import pandas as pd
# ==========

import metrics

with open('config.yaml', 'r') as f:
    config_data = yaml.safe_load(f)

model_dict = config_data['model_dict']

metrics_map = {
    # MRC
    "note_mrc": "calculate_semantic_similarity",
    
    # Hash-tag
    "note_hashtag_single": "calculate_answer_accuracy",
    "note_hashtag_multi": "calculate_multiple_choice_f1",
    
    # Note-Taxonomy
    "note_taxonomy_one_level": "calculate_three_level_accuracy",
    "note_taxonomy_three_levels": "calculate_answer_accuracy",

    # Note-gender
    "note_gender": "calculate_answer_accuracy",
    
    # Query-corr
    "note_querycorr_two_levels": "calculate_answer_accuracy",
    "note_querycorr_five_levels": "calculate_answer_accuracy",
    
    # Note-comment
    "note_comment_primary": "calculate_answer_accuracy",
    "note_comment_sub_level": "calculate_answer_accuracy",

    # Note-OCR
    "note_ocr": "calculate_ocr_metrics",

    # Note-Query_Gen
    "note_query_gen": "calculate_answer_accuracy",
}

location_map = {
    # MRC
    "note_mrc": "note_mrc",
    
    # Hash-tag
    "note_hashtag_single": "note_hashtag_single",
    "note_hashtag_multi": "note_hashtag_multi",
    
    # Note-Taxonomy
    "note_taxonomy_one_level": "note_taxonomy_one_level",
    "note_taxonomy_three_levels": "note_taxonomy_three_levels",

    # Note-gender
    "note_gender": "note_gender",
    
    # Query-corr
    "note_querycorr_two_levels": "note_querycorr_two_levels",
    "note_querycorr_five_levels": "note_querycorr_five_levels",
    
    # Note-comment
    "note_comment_primary": "note_comment_primary",
    "note_comment_sub_level": "note_comment_sub_level",

    # Note-OCR
    "note_ocr": "note_ocr",

    # Note-Query_Gen
    "note_query_gen": "note_query_gen",
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)

def count_all_dataset_result(location_map, model_path):
    
    final_result = {}

    for task_name, task_path in location_map.items():
        task_abs_path = Path(model_path) / task_path

        with open(task_abs_path / "data.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        metric_func = getattr(metrics, metrics_map[task_name])

        eval_result = 0
        
        for item in tqdm.tqdm(data, desc=f"开始计算{task_name}, 计算方法{metrics_map[task_name]}"):
            # try:
                eval_result += metric_func(item)
            # except Exception as e:
            #     print(e)
            #     print(f"失败 {model_path}: {task_name}")
            #     eval_result += 0

        eval_result /= len(data)

        final_result[task_name] = eval_result

    total_score = 0
    for task_name, task_path in location_map.items():
        total_score += final_result[task_name]
    
    total_score = total_score / len(location_map)
    final_result["total_score"] = total_score

    return final_result

        
def main():
    result_df = {}
    
    for model_name, model_path in model_dict.items():
        print(f"开始计算模型：{model_name}")
        result = count_all_dataset_result(location_map, model_path)
        result_df[model_name] = result
    
    df = pd.DataFrame(result_df)
    df.to_csv("result.csv")

    

if __name__ == "__main__":
    import metrics
    main()
    # check_none_main() 
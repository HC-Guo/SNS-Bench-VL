from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util
import yaml

# 加载配置文件
with open('config.yaml', 'r') as f:
    config_data = yaml.safe_load(f)

# 初始化BGE模型
semantic_model = SentenceTransformer(config_data['bge_path'])


def calculate_answer_accuracy(data_item):
    """
    计算答案完全匹配准确率
    """
    ground_truth = data_item['conversations'][1]['value']
    model_output = data_item['model_result']
    return 1 if model_output in ground_truth else 0


def calculate_three_level_accuracy(data_item):
    """
    计算三级准确率（词级别匹配）
    """
    ground_truth = data_item['conversations'][1]['value'].split(' ')
    model_output = data_item['model_result'].split(' ')
    
    match_count = 0
    max_length = min(len(ground_truth), len(model_output))
    
    for i in range(max_length):
        if ground_truth[i] in model_output[i]:
            match_count += 1
    
    return match_count / 3 if max_length >= 3 else 0


def calculate_multiple_choice_f1(data_item):
    """
    计算多选题F1分数
    """
    # 处理标准答案
    reference_answers = data_item['conversations'][1]['value'].split(' ')
    reference_set = set(reference_answers)
    
    # 处理模型预测结果
    try:
        predicted_answers = data_item['model_result'].split(' ')
    except Exception as e:
        print(f"处理模型结果时出错: {e}")
        predicted_answers = []
        
    predicted_set = set(predicted_answers)
    
    # 计算TP、FP、FN
    true_positives = len(reference_set & predicted_set)
    false_positives = len(predicted_set - reference_set)
    false_negatives = len(reference_set - predicted_set)
    
    # 计算精确率、召回率和F1分数
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1_score


def calculate_semantic_similarity(data_item):
    """
    计算答案和模型输出的语义相似度分数
    """
    reference_text = data_item['conversations'][1]['value']
    predicted_text = data_item['model_result']
    
    # 生成向量表示
    embedding1 = semantic_model.encode(reference_text, normalize_embeddings=True)
    embedding2 = semantic_model.encode(predicted_text, normalize_embeddings=True)
    
    # 计算余弦相似度
    similarity_score = float(util.cos_sim(embedding1, embedding2).item())
    
    return similarity_score


def calculate_ocr_metrics(data_item, return_detailed=False):
    """
    计算答案质量的综合评估指标（ROUGE、BLEU和BGE语义相似度）
    """
    reference_text = data_item['conversations'][1]['value']
    predicted_text = data_item['model_result']
    
    # 1. 计算ROUGE分数
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference_text, predicted_text)
    rouge_1 = rouge_scores['rouge1'].fmeasure
    rouge_l = rouge_scores['rougeL'].fmeasure
    avg_rouge = (rouge_1 + rouge_l) / 2
    
    # 2. 计算BLEU分数
    reference_tokens = [list(reference_text)]
    predicted_tokens = list(predicted_text)
    bleu_score = sentence_bleu(
        reference_tokens, 
        predicted_tokens, 
        smoothing_function=SmoothingFunction().method1
    )
    
    # 3. 计算BGE语义相似度
    embedding1 = semantic_model.encode(reference_text, normalize_embeddings=True)
    embedding2 = semantic_model.encode(predicted_text, normalize_embeddings=True)
    semantic_score = float(util.cos_sim(embedding1, embedding2).item())
    
    # 计算综合得分
    overall_score = (avg_rouge + bleu_score + semantic_score) / 3.0
    
    if return_detailed:
        return {
            'average_rouge': avg_rouge,
            'bleu_score': bleu_score,
            'semantic_similarity': semantic_score,
            'overall_quality': overall_score
        }
    else:
        return overall_score
import numpy as np

def precision_at_k(predicted_ids: list, ground_truth_set: set, k: int) -> float:

    top_k_preds = predicted_ids[:k]
    if not top_k_preds or k == 0:
        return 0.0

    hits = sum(1 for doc_id in top_k_preds if doc_id in ground_truth_set)
    return hits / k

def recall_at_k(predicted_ids: list, ground_truth_set: set, k: int) -> float:

    top_k_preds = predicted_ids[:k]
    if not ground_truth_set or k == 0:
        return 0.0

    hits = sum(1 for doc_id in top_k_preds if doc_id in ground_truth_set)
    return hits / len(ground_truth_set)

def mrr_at_10(predicted_ids: list, ground_truth_set: set) -> float:

    for rank, doc_id in enumerate(predicted_ids[:10]):
        if doc_id in ground_truth_set:
            return 1.0 / (rank + 1)
    return 0.0

def map_at_100(predicted_ids: list, ground_truth_set: set) -> float:

    if not ground_truth_set:
        return 0.0

    running_precision_sum = 0.0
    actual_hits = 0

    for rank, doc_id in enumerate(predicted_ids[:100]):
        if doc_id in ground_truth_set:
            actual_hits += 1
            precision_at_i = actual_hits / (rank + 1)
            running_precision_sum += precision_at_i

    return running_precision_sum / len(ground_truth_set)

def ndcg_at_10(predicted_ids: list, ground_truth_labels: dict) -> float:

    top_10_preds = predicted_ids[:10]
    if not top_10_preds or not ground_truth_labels:
        return 0.0

    dcg = 0.0
    for rank, doc_id in enumerate(top_10_preds):
        relevance = ground_truth_labels.get(doc_id, 0)
        dcg += relevance / np.log2(rank + 2)

    true_scores = sorted(ground_truth_labels.values(), reverse=True)[:10]
    idcg = 0.0

    for rank, relevance in enumerate(true_scores):
        idcg += relevance / np.log2(rank + 2)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg

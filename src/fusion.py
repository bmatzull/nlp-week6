def reciprocal_rank_fusion(sparse_results: list, dense_results: list, k_constant: int = 60, top_n: int = 10) -> list:
    rrf_scores = {}

    for rank, doc_id in enumerate(sparse_results):
        rank_position = rank + 1
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_constant + rank_position))

    for rank, doc_id in enumerate(dense_results):
        rank_position = rank + 1
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k_constant + rank_position))

    sorted_predictions = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return [doc_id for doc_id, score in sorted_predictions[:top_n]]


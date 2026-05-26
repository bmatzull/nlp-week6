import sys
import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_ms_marco_evaluation_set, load_scifact_evaluation_set
from src.sparse_models import SparseRetrievalEngine
from src.dense_models import DenseRetrievalEngine, CrossEncoderReRanker, HyDEEngine
from src.fusion import reciprocal_rank_fusion
from src.late_interaction import ColBERTRetriever
from src.metrics import (
    precision_at_k,
    recall_at_k,
    mrr_at_10,
    map_at_100,
    ndcg_at_10
)


def evaluate_retrieval_method(queries, corpus, qrels, retrieval_func, method_tag: str):
    p1, p5, p10 = [], [], []
    r10, r50, r100 = [], [], []
    mrrs, maps, ndcgs = [], [], []
    latencies = []

    target_queries = queries
    if "M8_HyDE" in method_tag:
        target_queries = queries[:10]

    for q_item in tqdm(target_queries, desc=f"Processing {method_tag}"):
        q_id = q_item["query_id"]
        q_text = q_item["query_text"]

        gt_set = qrels.get(q_id, set())

        if isinstance(gt_set, dict):
            gt_dict = gt_set
            gt_set_for_recall = set(gt_dict.keys())
        else:
            gt_dict = {doc_id: 1 for doc_id in gt_set}
            gt_set_for_recall = gt_set

        start_time = time.perf_counter()
        try:
            predicted_ids = retrieval_func(q_text)
        except Exception as e:
            predicted_ids = []
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000.0
        latencies.append(latency_ms)

        if len(predicted_ids) < 100:
            predicted_ids = list(predicted_ids) + ["PADDING_ID"] * (100 - len(predicted_ids))

        p1.append(precision_at_k(predicted_ids, gt_set_for_recall, k=1))
        p5.append(precision_at_k(predicted_ids, gt_set_for_recall, k=5))
        p10.append(precision_at_k(predicted_ids, gt_set_for_recall, k=10))

        r10.append(recall_at_k(predicted_ids, gt_set_for_recall, k=10))
        r50.append(recall_at_k(predicted_ids, gt_set_for_recall, k=50))
        r100.append(recall_at_k(predicted_ids, gt_set_for_recall, k=100))

        mrrs.append(mrr_at_10(predicted_ids, gt_set_for_recall))
        maps.append(map_at_100(predicted_ids, gt_set_for_recall))
        ndcgs.append(ndcg_at_10(predicted_ids, gt_dict))

    return {
        "Method": method_tag,
        "P@1": round(np.mean(p1), 4),
        "P@5": round(np.mean(p5), 4),
        "P@10": round(np.mean(p10), 4),
        "R@10": round(np.mean(r10), 4),
        "R@50": round(np.mean(r50), 4),
        "R@100": round(np.mean(r100), 4),
        "MRR@10": round(np.mean(mrrs), 4),
        "MAP@100": round(np.mean(maps), 4),
        "NDCG@10": round(np.mean(ndcgs), 4),
        "Latency (ms)": round(np.mean(latencies), 2)
    }


def execute_dataset_benchmark(dataset_name: str):
    if dataset_name.lower() == "ms_marco":
        queries, corpus, qrels = load_ms_marco_evaluation_set(sample_size=300)
    else:
        queries, corpus, qrels = load_scifact_evaluation_set(sample_size=300)

    m1_engine = SparseRetrievalEngine(method_type="BM25")
    m1_engine.fit(corpus)

    m2_engine = SparseRetrievalEngine(method_type="TFIDF")
    m2_engine.fit(corpus)

    m3_engine = DenseRetrievalEngine(model_name="all-MiniLM-L6-v2", dataset_tag=dataset_name)
    m3_engine.fit(corpus)

    m4_engine = DenseRetrievalEngine(model_name="msmarco-distilbert-base-v3", dataset_tag=dataset_name)
    m4_engine.fit(corpus)

    m6_reranker = CrossEncoderReRanker()
    m8_hyde = HyDEEngine(dense_base_engine=m3_engine)

    m7_colbert = ColBERTRetriever()
    m7_colbert.fit(corpus)

    retrieval_functions = {
        "M1_BM25": lambda q: m1_engine.query(q, top_k=100),
        "M2_TFIDF": lambda q: m2_engine.query(q, top_k=100),
        "M3_Dense_General": lambda q: m3_engine.query(q, top_k=100),
        "M4_Dense_Domain": lambda q: m4_engine.query(q, top_k=100),
        "M5_Hybrid_RRF": lambda q: reciprocal_rank_fusion(m1_engine.query(q, top_k=100), m3_engine.query(q, top_k=100)),
        "M6_CrossEncoder": lambda q: m6_reranker.re_rank(q, m3_engine.query(q, top_k=100), corpus),
        "M7_ColBERTv2": lambda q: m7_colbert.query(q, top_k=10),
        "M8_HyDE": lambda q: m8_hyde.query_with_hyde(q, top_k=10)
    }

    dataset_rows = []
    for method_tag, query_func in retrieval_functions.items():
        metrics_summary = evaluate_retrieval_method(queries, corpus, qrels, query_func, method_tag)
        dataset_rows.append(metrics_summary)

    df_results = pd.DataFrame(dataset_rows)
    os.makedirs("results", exist_ok=True)
    df_results.to_csv(f"results/{dataset_name.lower()}_results.csv", index=False)

    print(f"\nResults for {dataset_name.upper()}:")
    print(df_results.to_string(index=False))


def main():
    execute_dataset_benchmark("ms_marco")
    execute_dataset_benchmark("scifact")

if __name__ == "__main__":
    main()
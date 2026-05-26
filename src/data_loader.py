import random
from datasets import load_dataset

def sample_dataset_elements(dataset_split, sample_size: int = 300, seed: int = 42):

    random.seed(seed)

    total_rows = len(dataset_split)
    target_size = min(sample_size, total_rows)
    sampled_indices = random.sample(range(total_rows), target_size)

    return dataset_split.select(sampled_indices)

def load_ms_marco_evaluation_set(sample_size: int = 300):

    dataset = load_dataset("microsoft/ms_marco", "v1.1", split="validation")
    sampled_data = sample_dataset_elements(dataset, sample_size=sample_size)

    queries = []
    corpus = {}
    qrels = {}

    for row in sampled_data:
        q_id = str(row["query_id"])
        if not row["query"]:
            continue
        q_text = row["query"]

        queries.append({"query_id": q_id, "query_text": q_text})
        passages = row["passages"]
        relevant_docs = set()

        for idx, p_text in enumerate(passages["passage_text"]):
            d_id = f"ms_marco_{q_id}_{idx}"
            is_selected = passages["is_selected"][idx]

            corpus[d_id] = p_text
            if is_selected == 1:
                relevant_docs.add(d_id)

        qrels[q_id] = relevant_docs

    return queries, corpus, qrels

def load_scifact_evaluation_set(sample_size: int = 300):
    corpus_ds = load_dataset("mteb/scifact", "corpus", split="corpus")
    queries_ds = load_dataset("mteb/scifact", "queries", split="queries")
    qrels_ds = load_dataset("mteb/scifact", "default", split="test")

    sampled_queries_ds = sample_dataset_elements(queries_ds, sample_size=sample_size)
    corpus = {str(row["_id"]): f"{row.get('title', '')} {row.get('text', '')}".strip() for row in corpus_ds}

    sampled_query_ids = set()
    queries = []
    for row in sampled_queries_ds:
        q_id = str(row["_id"])
        queries.append({"query_id": q_id, "query_text": row["text"]})
        sampled_query_ids.add(q_id)

    qrels = {}
    for row in qrels_ds:
        q_id = str(row["query-id"])
        if q_id in sampled_query_ids:
            d_id = str(row["corpus-id"])
            score = int(row["score"])
            if q_id not in qrels:
                qrels[q_id] = {}
            qrels[q_id][d_id] = score

    return queries, corpus, qrels

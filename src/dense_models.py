import os
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

class DenseRetrievalEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dataset_tag: str = "default"):
        self.model_name = model_name
        self.dataset_tag = dataset_tag
        self.encoder = SentenceTransformer(model_name)
        self.corpus_ids = []
        self.corpus_embeddings = None

        safe_model_string = model_name.replace("/", "_")
        self.cache_dir = "data/cache"
        self.emb_cache_path = os.path.join(self.cache_dir, f"emb_{safe_model_string}_{dataset_tag}.npy")
        self.id_cache_path = os.path.join(self.cache_dir, f"ids_{safe_model_string}_{dataset_tag}.txt")

    def fit(self, corpus_dict: dict):
        self.corpus_ids = list(corpus_dict.keys())
        raw_documents = list(corpus_dict.values())

        if os.path.exists(self.emb_cache_path) and os.path.exists(self.id_cache_path):
            self.corpus_embeddings = np.load(self.emb_cache_path)
            with open(self.id_cache_path, "r", encoding="utf-8") as f:
                self.corpus_ids = [line.strip() for line in f.readlines()]
            return

        self.corpus_embeddings = self.encoder.encode(
            raw_documents,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        os.makedirs(self.cache_dir, exist_ok=True)
        np.save(self.emb_cache_path, self.corpus_embeddings)
        with open(self.id_cache_path, "w", encoding="utf-8") as f:
            for doc_id in self.corpus_ids:
                f.write(f"{doc_id}\n")

    def query(self, query_text: str, top_k: int = 100) -> list:
        if self.corpus_embeddings is None:
            return []

        query_vector = self.encoder.encode([query_text], convert_to_numpy=True)
        scores = np.dot(self.corpus_embeddings, query_vector.T).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        return [self.corpus_ids[idx] for idx in top_indices]

class CrossEncoderReRanker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def re_rank(self, query_text: str, candidate_ids: list, corpus_dict: dict, top_n: int = 10) -> list:
        if not candidate_ids:
            return []

        pairs = [[query_text, corpus_dict[doc_id]] for doc_id in candidate_ids]
        scores = self.model.predict(pairs)

        ranked_pairs = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, score in ranked_pairs[:top_n]]

class HyDEEngine:
    def __init__(self, dense_base_engine: DenseRetrievalEngine):
        self.base_engine = dense_base_engine

    def query_with_hyde(self, query_text: str, top_k: int = 10) -> list:
        pseudo_response = (
            f"Regarding the legal inquiry or technical baseline: '{query_text}'. "
            f"The verified structural provisions and analytical details point to case files "
            f"and matching references verifying this specific condition sequence."
        )
        return self.base_engine.query(pseudo_response, top_k=top_k)
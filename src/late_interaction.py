import torch
from transformers import AutoTokenizer, AutoModel

class ColBERTRetriever:
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        print(f"Loading ColBERTv2 Late Interaction Model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        self.corpus_ids = []
        self.corpus_token_embeddings = []

    def fit(self, corpus_dict: dict):
        self.corpus_ids = list(corpus_dict.keys())
        for doc_text in corpus_dict.values():
            inputs = self.tokenizer(doc_text, return_tensors="pt",
                                    truncation=True, max_length=128, padding=True)
            with torch.no_grad():
                output = self.model(**inputs)
            self.corpus_token_embeddings.append(output.last_hidden_state.squeeze(0))

    def query(self, query_text: str, top_k: int = 10) -> list:
        inputs = self.tokenizer(query_text, return_tensors="pt",
                                truncation=True, max_length=32)
        with torch.no_grad():
            output = self.model(**inputs)
        query_embeddings = output.last_hidden_state.squeeze(0)  # [q_tokens, dim]

        scores = []
        for doc_embeddings in self.corpus_token_embeddings:
            sim_matrix = torch.matmul(query_embeddings, doc_embeddings.T)
            max_sims = sim_matrix.max(dim=1).values  # [q_tokens]
            scores.append(max_sims.sum().item())

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.corpus_ids[i] for i in top_indices]
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

class SparseRetrievalEngine:
    def __init__(self, method_type: str = "BM25"):
        self.method_type = method_type.upper()
        self.corpus_ids = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.bm25 = None

    def fit(self, corpus_dict: dict):
        self.corpus_ids = list(corpus_dict.keys())
        raw_documents = list(corpus_dict.values())

        if self.method_type == "TFIDF":
            self.vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
            self.tfidf_matrix = self.vectorizer.fit_transform(raw_documents)
        elif self.method_type == "BM25":
            tokenized_corpus = [doc.lower().split() for doc in raw_documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            raise NotImplementedError(self.method_type)

    def query(self, query_text: str, top_k: int = 100) -> list:
        if not self.corpus_ids:
            return []

        if self.method_type == "TFIDF":
            query_vector = self.vectorizer.transform([query_text])
            scores = (self.tfidf_matrix * query_vector.T).toarray().flatten()
            top_indices = scores.argsort()[::-1][:top_k]
            return [self.corpus_ids[idx] for idx in top_indices]

        elif self.method_type == "BM25":
            tokenized_query = query_text.lower().split()
            scores = self.bm25.get_scores(tokenized_query)
            top_indices = scores.argsort()[::-1][:top_k]
            return [self.corpus_ids[idx] for idx in top_indices]

        return []


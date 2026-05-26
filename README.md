#  Retrieval Benchmark: NLP Week 6

## Project Structure

```
nlp-week6/
├── experiments/
│   ├── data/
│   └── cache/ 
│   └── run_benchmark.py       
│   └── results/
│       ├── ms_marco_results.csv
│       └── scifact_results.csv
├── src/
│   ├── __init__.py
│   ├── data_loader.py         
│   ├── sparse_models.py        
│   ├── dense_models.py        
│   ├── late_interaction.py     
│   ├── fusion.py               
│   └── metrics.py              
├── requirements.txt
└── README.md
```

## Setup

```bash

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Benchmark

```bash

python experiments/run_benchmark.py
```

## The 8 Methods

| ID | Method | Description |
|----|--------|-------------|
| M1 | BM25 | Sparse keyword retrieval using Okapi BM25 |
| M2 | TF-IDF | Sparse retrieval using TF-IDF cosine similarity |
| M3 | Dense General | Bi-encoder using `all-MiniLM-L6-v2` |
| M4 | Dense Domain | Bi-encoder using `msmarco-distilbert-base-v3` |
| M5 | Hybrid RRF | Reciprocal Rank Fusion of M1 + M3 |
| M6 | Cross-Encoder | M3 retrieval + `cross-encoder/ms-marco-MiniLM-L-6-v2` re-ranking |
| M7 | ColBERTv2 | Late interaction (approximated via M3 dense encoder) |
| M8 | HyDE | Hypothetical Document Embeddings using a template pseudo-document + M3 |


## Datasets

**MS MARCO v1.1** (`microsoft/ms_marco`, `validation` split)
- General-domain web search queries
- 1 relevant document per query
- Primary metric: MRR@10

**SciFact** (`mteb/scifact`)
- Scientific claim verification against research paper abstracts
- Graded relevance labels
- Primary metric: NDCG@10


## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| P@1, P@5, P@10 | Fraction of relevant documents in the top k results |
| R@10, R@50, R@100 | Fraction of all relevant documents retrieved in top k |
| MRR@10 | Reciprocal rank of the first relevant document (top 10) |
| MAP@100 | Mean average precision across relevant documents in top 100 |
| NDCG@10 | Normalized discounted cumulative gain at 10 (accounts for graded relevance) |
| Latency (ms) | Average query time in milliseconds |


## Results

### MS MARCO

| Method | P@1 | P@5 | P@10 | R@10 | R@50 | R@100 | MRR@10 | MAP@100 | NDCG@10 | Latency (ms) |
|--------|-----|-----|------|------|------|-------|--------|--------|--------|--------------|
| M1_BM25 | 0.2500 | 0.1493 | 0.0890 | 0.8400 | 0.9267 | 0.9300 | 0.4316 | 0.4336 | 0.5289 | 1.64 |
| M2_TFIDF | 0.2333 | 0.1407 | 0.0930 | 0.8700 | 0.9300 | 0.9300 | 0.4200 | 0.4210 | 0.5270 | 0.41 |
| M3_Dense_General | 0.3600 | 0.1780 | 0.1023 | 0.9567 | 0.9633 | 0.9633 | 0.5529 | 0.5494 | 0.6498 | 4.84 |
| M4_Dense_Domain | 0.3833 | 0.1893 | 0.1027 | 0.9600 | 0.9633 | 0.9633 | 0.5856 | 0.5808 | 0.6759 | 10.58 |
| M5_Hybrid_RRF | 0.3233 | 0.1707 | 0.0990 | 0.9283 | 0.9283 | 0.9283 | 0.5193 | 0.5159 | 0.6178 | 6.98 |
| M6_CrossEncoder | 0.4667 | 0.1927 | 0.1023 | 0.9567 | 0.9567 | 0.9567 | 0.6388 | 0.6375 | 0.7173 | 734.58 |
| M7_ColBERTv2 | 0.1967 | 0.1227 | 0.0850 | 0.7933 | 0.7933 | 0.7933 | 0.3646 | 0.3593 | 0.4647 | 92.59 |
| M8_HyDE | 0.2000 | 0.1200 | 0.0800 | 0.7500 | 0.7500 | 0.7500 | 0.3917 | 0.3854 | 0.4771 | 7.16 |

### SciFact

| Method | P@1     | P@5 | P@10 | R@10 | R@50 | R@100 | MRR@10 | MAP@100 | NDCG@10 | Latency (ms) |
|--------|---------|-----|------|------|------|-------|--------|--------|--------|--------------|
| M1_BM25 | 0.1100  | 0.0333 | 0.0183 | 0.1606 | 0.1806 | 0.1806 | 0.1281 | 0.1248 | 0.1338 | 9.19 |
| M2_TFIDF | 0.0900  | 0.0327 | 0.0187 | 0.1596 | 0.2030 | 0.2093 | 0.1139 | 0.1101 | 0.1214 | 1.15 |
| M3_Dense_General | 0.1100  | 0.0413 | 0.0233 | 0.1947 | 0.2120 | 0.2267 | 0.1392 | 0.1362 | 0.1510 | 5.54 |
| M4_Dense_Domain | 0.0867  | 0.0307 | 0.0183 | 0.1494 | 0.1926 | 0.1956 | 0.1074 | 0.1021 | 0.1135 | 12.71 |
| M5_Hybrid_RRF | 0.1200  | 0.0353 | 0.0213 | 0.1839 | 0.1839 | 0.1839 | 0.1424 | 0.1387 | 0.1509 | 19.43 |
| M6_CrossEncoder | 0.1367  | 0.0413 | 0.0230 | 0.1919 | 0.1919 | 0.1919 | 0.1588 | 0.1518 | 0.1637 | 2620.83 |
| M7_ColBERTv2 | 0.0967  | 0.0273 | 0.0150 | 0.1356 | 0.1356 | 0.1356 | 0.1106 | 0.1063 | 0.1145 | 232.98 |
| M8_HyDE | 0.1000  | 0.0200 | 0.0100 | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 0.1000 | 7.69 |


## Analysis

### Which method wins overall, and does it depend on the metric?

**M6 (Cross-Encoder) wins on nearly every metric on MS MARCO**: 


### Where does BM25 beat dense retrieval?

On MS MARCO, BM25 (M1) actually beats dense methods on **R@50 and R@100** compared to M5 Hybrid RRF. BM25 tends to 
win on queries with specific terms, where a dense model may not have enough data during training. But on SciFact it loses 
to M3, which suggests that it struggles when scientific language requires semantic understanding.


### Top 5 queries where best and worst methods disagree most

1. **Paraphrase queries**: the document uses different wording than the query (dense wins, BM25 fails)
2. **Exact term queries**: the query contains a rare proper noun or acronym (BM25 wins, dense may miss)
3. **Short ambiguous queries**: single-word queries where context is needed (cross-encoder wins by considering full document context)
4. **Scientific claims**: where the relevant document refutes rather than confirms the query (all methods struggle)
5. **Long descriptive queries**: where BM25 over-weights repeated terms but dense captures the overall intent


### Where does HyDE win and fail?

HyDE is limited to 10 queries in this benchmark and uses a template-based pseudo-document rather than a real LLM, 
which limits its effectiveness. But it could win where Queries are phrased as questions, since the generated pseudo-document
better resembles a relevant passage than the query instead. Problems are that the hardcoded template is generic abd does 
not adapt to the query content and with only 10 evaluation queries, results are noisy and not really representative.


### Does domain-specific M4 outperform general M3? On which dataset?

Yes, on MS MARCO it outperforms M3 on every metric, since M4 was fine-tuned specifically on MS MARCO data. On SciFact, 
M4 underperforms M3 across all metrics. 


### Does hybrid RRF (M5) beat both individual methods?

On MS MARCO, M5 beats BM25, but doesn't beat M3. 
On SciFact, M5 beats BM25 and TF-IDF on most metrics, but doesn't beat M3 alone.


### What does re-ranking (M6) cost in latency vs. what it gains? Is it worth it?

**MS MARCO:**
- Latency cost: 734ms vs 4.84ms for M3 — roughly
- NDCG@10 gain: 0.7173 vs 0.6498
- MRR@10 gain: 0.6388 vs 0.5529

**SciFact:**
- Latency cost: 2620ms vs 5.54ms for M3
- NDCG@10 gain: 0.1637 vs 0.1510

For evaluations, where accuracy matters most, M6 is the best method, but for real-time systems, the latency is too
high for most applications.
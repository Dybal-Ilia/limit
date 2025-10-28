# LIMIT: Learning Information Retrieval Models with Intelligent Testing

A comprehensive information retrieval research framework implementing and evaluating multiple retrieval approaches including BM25, dense retrieval, sparse retrieval (SPLADE), and multi-vector late-interaction models with advanced hybrid fusion and reranking capabilities.

## Overview

LIMIT is designed for information retrieval research and experimentation, providing implementations of state-of-the-art retrieval models with comprehensive evaluation metrics. The framework supports sophisticated hybrid retrieval pipelines that combine multiple approaches using configurable score fusion strategies, and includes robust ablation study capabilities for systematic analysis of embedding dimensionality impact on retrieval performance.

## Features

- **Multi-Modal Retrieval**: Support for BM25 (lexical), dense retrieval (sentence transformers), and sparse retrieval (SPLADE)
- **Hybrid Retrieval Pipeline**: Score-level fusion combining multiple retrieval approaches
  - **Fusion strategies**: Weighted-sum (default) or Reciprocal Rank Fusion (RRF)
  - **Score normalization**: Min-max, z-score (sigmoid), or rank-based normalization
  - **Flexible weighting**: Per-retriever weights to emphasize different signal sources
- **Cross-Encoder Reranking**: Two-stage retrieval with transformer-based cross-encoder reranking
- **Multi-Vector Retrieval**: Optional ColBERT-style late interaction retriever for token-level matching
- **Query Planning**: LLM-based query decomposition for complex multi-faceted queries
- **Comprehensive Evaluation**: Standard IR metrics including Recall@K and NDCG@K at multiple cutoffs
- **Ablation Studies**: Systematic analysis of embedding dimension impact with per-dimension document re-encoding
- **Configurable Experiments**: YAML-based configuration for all pipeline components
- **Analysis Notebook**: Jupyter notebook for experiment analysis, visualization, and ablation diagnostics


## Installation

### Requirements

- Python >= 3.12
- CUDA-capable GPU (recommended for dense retrieval and transformers)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Dybal-Ilia/limit.git
cd limit
```

2. Install dependencies using `uv` (recommended) or `pip`:
```bash
# Using uv
uv sync

# Or using pip
pip install -e .
```

### Dependencies

Core dependencies (auto-installed):
- `numpy`: Numerical computations and array operations
- `torch`: PyTorch for neural models and GPU acceleration
- `transformers`: Hugging Face transformers for SPLADE and reranking
- `sentence-transformers`: Dense retrieval embedding models
- `rank-bm25`: Efficient BM25 implementation
- `pyyaml`: Configuration file parsing
- `pandas`: Results analysis and CSV handling
- `matplotlib`: Visualization and plotting
- `ipykernel`: Jupyter notebook support

## Usage

### Quick Start

Run the main experiment pipeline:

```bash
python main.py
```

This will execute the full baseline experiment using the configuration in `src/config/config.yaml`, including:
- Multi-modal retrieval (BM25 + SPLADE + Dense)
- Optional query planning and decomposition
- Cross-encoder reranking
- Evaluation on specified queries
- Optional ablation studies across embedding dimensions

### Running Ablation Studies

Run ablation experiments separately:

```bash
python run_ablation.py
```

This executes systematic dimensionality ablation studies as configured in `config.yaml`, generating per-dimension CSV results in the `runs/` directory.

### Analysis Notebook

Explore results interactively:

```bash
jupyter notebook src/notebooks/limit_analysis.ipynb
```

The notebook includes:
- Environment setup and data loading
- Baseline experiment execution
- Fusion strategy comparison
- Ablation study visualization
- Query planning demonstrations
- Error analysis and diagnostics

### Configuration

Edit `src/config/config.yaml` to customize all aspects of the pipeline:

#### Retrieval Settings
```yaml
retriever:
  top_k: 500              # Number of candidates to retrieve
  rerank_k: 200           # Number of documents to rerank
  fusion: "weighted"      # Fusion strategy: 'weighted' or 'rrf'
  normalizer: "minmax"    # Score normalization: 'minmax', 'zscore', or 'rank'
  rrf_k: 60               # RRF parameter (when fusion='rrf')
  weights: [1.0, 1.0, 1.0]  # Per-retriever weights [bm25, splade, dense]
```

#### Model Configuration
```yaml
models:
  bm25: "bm25"
  splade: "naver/splade-cocondenser-ensembledistil"
  dense: "sentence-transformers/all-mpnet-base-v2"
  reranker: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  # Optional: enable multi-vector retrieval
  # late_interaction: "sentence-transformers/all-MiniLM-L6-v2"
```

#### Ablation Studies
```yaml
ablation:
  enabled: true           # Enable/disable ablation experiments
  debug: true             # Print detailed shape/vector info
  mode: "truncate"        # 'truncate' (reduce dims) or 'pad' (extend dims)
  dims: [32, 64, 128, 256, 512, 768]  # Dimensions to test
  output: "runs/ablation_results.csv"
```

#### Query Planning
```yaml
query_planning:
  enabled: true           # Enable LLM-based query decomposition
  model: "sshleifer/tiny-gpt2"  # Language model for decomposition
  max_new_tokens: 64      # Max tokens for generated sub-queries
  merge: "union"          # Merge strategy: 'union' or 'rrf'
  rrf_k: 60               # RRF parameter for merging
```

#### Data Paths
```yaml
paths:
  corpus: "src/data/corpus-small.jsonl"
  queries: "src/data/queries-small.jsonl"
  qrels: "src/data/qrels-small.jsonl"
```

### Data Format

The framework expects JSONL format for all data files:

**Corpus** (`corpus.jsonl`):
```json
{"_id": "doc1", "text": "Document content..."}
```

**Queries** (`queries.jsonl`):
```json
{"_id": "query1", "text": "Query text..."}
```

**Relevance Judgments** (`qrels.jsonl`):
```json
{"query-id": "query1", "corpus-id": "doc1", "score": 1}
```

## Models

### BM25 Retriever (`bm25.py`)
Traditional term-frequency based retrieval using the Okapi BM25 algorithm. Provides strong baseline performance for keyword matching without requiring neural models or GPU computation.

**Key features:**
- Classic probabilistic ranking function
- No training required
- Fast CPU-based retrieval
- Excellent for lexical matching

### Dense Retriever (`dense.py`)
Utilizes sentence transformer models to encode queries and documents into dense vector representations (typically 768-dimensional). 

**Key features:**
- Semantic similarity via cosine similarity
- Automatic GPU/CPU device selection
- Batch processing for efficiency
- L2 normalization of embeddings
- Captures semantic relationships beyond keyword matching

### SPLADE Retriever (`splade.py`)
Implements sparse lexical retrieval using learned sparse representations. Based on transformer models that output vocabulary-sized sparse vectors (30K+ dimensions) for improved semantic matching while maintaining interpretability.

**Key features:**
- Learned sparse representations
- Vocabulary-aligned dimensions
- Efficient inverted index retrieval possible
- Combines neural understanding with sparse efficiency

### Hybrid Retriever (`hybrid.py`)
**Core contribution:** Performs score-level fusion of multiple retrievers rather than naive embedding averaging.

**Key features:**
- Multiple score normalization strategies (min-max, z-score, rank-based)
- Two fusion modes:
  - **Weighted fusion**: Linear combination with per-retriever weights
  - **RRF (Reciprocal Rank Fusion)**: Rank-based fusion resistant to score scale differences
- Flexible weight configuration supporting 3+ retrievers
- Proper handling of heterogeneous score distributions

**Implementation note:** Earlier versions incorrectly averaged embeddings from different retrievers; current version performs proper score fusion after independent retrieval.

### Late Interaction Retriever (`late_interaction.py`)
Implements a simplified ColBERT-style retriever for multi-vector retrieval research.

**Key features:**
- Token-level encodings for queries and documents
- MaxSim scoring: `score(Q,D) = Σ_i max_j sim(q_i, d_j)`
- Captures fine-grained token-level interactions
- Experimental: useful for studying single-vector vs. multi-vector trade-offs

**Usage:** Enable via `models.late_interaction` in config. If present, adds a fourth candidate generator that can participate in hybrid fusion.

### Cross-Encoder Reranker (`reranker.py`)
Provides final ranking refinement using cross-encoder models that jointly encode query-document pairs for precise relevance scoring.

**Key features:**
- Joint query-document encoding (vs. independent encoding)
- Higher computational cost (applied to top-k only)
- State-of-the-art ranking quality
- Two-stage retrieval paradigm: fast candidate generation + precise reranking

## Evaluation Metrics

- **Recall@K**: Proportion of relevant documents retrieved in top-K results
- **NDCG@K**: Normalized Discounted Cumulative Gain considering ranking quality
- **Support for multiple K values**: Configurable evaluation depths (2, 10, 100)

## Ablation Studies

The framework includes sophisticated ablation testing capabilities to systematically study the impact of embedding dimensionality on retrieval performance.

### Features

- **Embedding Dimension Analysis**: Test performance across custom dimension ranges (e.g., 32, 64, 128, 256, 512, 768)
- **Two Operation Modes**:
  - `truncate`: Reduce dimensionality by removing higher dimensions (tests information loss)
  - `pad`: Extend dimensionality with zero-padding (tests robustness to dimension increase)
- **Proper Document Re-encoding**: Documents are re-encoded at each dimension (not just queries) to ensure vectors match the target dimensionality
- **Per-Query and Aggregated Results**: Detailed CSV outputs with both query-level and averaged metrics
- **Incremental Checkpointing**: Per-dimension CSV files saved during long-running experiments
- **Debug Mode**: Optional detailed logging of vector shapes and encoding info (set `ablation.debug: true`)

### Implementation Details

The ablation harness uses runtime monkeypatching to intercept `encode()` and `retrieve()` methods:
1. For each dimension `d`, wrap the original `encode()` to truncate/pad outputs to dimension `d`
2. Pre-compute document embeddings at dimension `d` for Dense and SPLADE retrievers
3. Wrap `retrieve()` methods to use the per-dimension document vectors
4. Run full retrieval pipeline at dimension `d` and record metrics
5. Restore original methods and proceed to next dimension

This ensures that both query and document vectors operate in the same reduced/extended dimensional space, providing accurate measurements of dimensionality impact.

### Output

Results are saved to `runs/ablation_results.csv` (or configured path) with columns:
- `mode`: truncate or pad
- `dim`: embedding dimension tested
- `recall@10`, `nDCG@10`: averaged metrics across all queries
- Per-query breakdowns available in `runs/ablation_dim_{d}.csv` files

### Known Limitations

The current implementation primarily affects Dense and SPLADE retrievers. BM25 is unaffected as it uses sparse term-frequency representations. Future work may explore dimensionality reduction techniques for SPLADE's vocabulary-aligned representations.

## Query Planning

Advanced query processing through LLM-based decomposition for handling complex multi-faceted queries.

### Features

- **Query Decomposition**: Breaks complex queries into manageable sub-components using language models
- **Flexible LM Backend**: Supports any Hugging Face transformers text-generation model
- **Candidate Merging**: Two strategies for combining sub-query results:
  - `union`: Simple union of all retrieved documents
  - `rrf`: Reciprocal Rank Fusion across sub-query rankings
- **Optional Execution**: Fully configurable via `query_planning.enabled` flag
- **Graceful Fallback**: If LLM initialization fails or is disabled, queries pass through unchanged

### Configuration

```yaml
query_planning:
  enabled: true
  model: "sshleifer/tiny-gpt2"  # Lightweight default; replace with stronger LM for production
  max_new_tokens: 64
  merge: "union"  # or 'rrf'
  rrf_k: 60
```

### Implementation

The `QueryPlanningAgent` uses a text-generation pipeline to decompose queries, then:
1. Retrieves candidates for each sub-query independently
2. Merges candidates using the configured strategy
3. Passes merged candidates to the reranker for final scoring

**Note:** The default model (`tiny-gpt2`) is chosen for fast prototyping. For production use, consider stronger instruction-tuned models like `meta-llama/Llama-2-7b-chat-hf` or similar.

## Logging

Comprehensive logging system provides detailed execution tracking:
- Query processing progress
- Retrieval statistics and diagnostics  
- Model performance metrics
- Debug information for troubleshooting

## Development

### Adding New Retrievers

1. Create a new file in `src/models/` (e.g., `my_retriever.py`)
2. Inherit from `BaseModel` in `src/models/base.py`
3. Implement required methods:
   ```python
   def encode(self, texts: List[str], **kwargs) -> np.ndarray:
       """Encode texts into embeddings/representations."""
       pass
   
   def retrieve(self, query: str, corpus: List[str], top_k: int = 100) -> List[Tuple[int, float]]:
       """Retrieve top-k documents for query. Returns list of (doc_idx, score) tuples."""
       pass
   ```
4. Register in `HybridRetrievalPipeline` or use standalone
5. Add configuration entry in `config.yaml` under `models`

### Custom Evaluation Metrics

Add new metrics to `src/evaluation/metrics.py`:

```python
def my_metric(predictions: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]], k: int = 10) -> Dict[str, float]:
    """
    predictions: {query_id: [doc_id1, doc_id2, ...]}
    qrels: {query_id: {doc_id: relevance_score}}
    Returns: {query_id: metric_value}
    """
    pass
```

Register in the evaluation loop in `baseline_experiment.py`.

### Extending Experiments

Create new experiment classes in `src/experiments/`:

```python
from src.experiments.baseline_experiment import BaselineExperiment

class MyExperiment(BaselineExperiment):
    def run(self):
        # Custom experiment logic
        pass
```

### Adding Fusion Strategies

Extend `HybridRetriever.retrieve()` in `src/models/hybrid.py`:

1. Add new normalization function to `_normalize_scores()`
2. Add new fusion strategy to the `if self.fusion == ...` block
3. Update config schema and documentation


## Research Applications

This framework is designed for:

- **Information Retrieval Research**: Comparative analysis of retrieval approaches (lexical vs. semantic vs. hybrid)
- **Ablation Studies**: Understanding the impact of embedding dimensionality on retrieval quality
- **Fusion Strategy Analysis**: Systematic comparison of score normalization and fusion techniques
- **Multi-Vector Retrieval**: Exploring ColBERT-style token-level matching vs. single-vector approaches
- **Query Planning**: Investigating query decomposition and multi-step retrieval strategies
- **Benchmark Evaluation**: Systematic evaluation on standard IR datasets (MS MARCO, BEIR, etc.)
- **Model Development**: Prototyping and testing new retrieval and reranking techniques
- **Hybrid System Design**: Exploring combinations of lexical, sparse, and dense retrieval methods

### Research Questions Supported

1. **How does embedding dimensionality affect retrieval performance?**
   - Use ablation studies with truncate/pad modes across dimension ranges
   - Analyze trade-offs between model size and retrieval quality

2. **Which fusion strategy works best for combining heterogeneous retrievers?**
   - Compare weighted fusion vs. RRF
   - Test different score normalization approaches (minmax, zscore, rank)

3. **Do multi-vector representations improve over single-vector dense retrieval?**
   - Compare LateInteractionRetriever vs. DenseRetriever
   - Analyze computational cost vs. quality trade-offs

4. **How effective is query planning for complex information needs?**
   - Enable/disable query planning and compare metrics
   - Analyze per-query impact of decomposition

## Performance Considerations

- **GPU Acceleration**: Automatic CUDA utilization for dense retrieval, SPLADE, and cross-encoder reranking
- **Batch Processing**: Efficient batch encoding for large document collections
- **Memory Management**: 
  - Documents encoded once and cached in memory
  - Streaming data loading for large corpora (JSONL format)
  - Ablation studies pre-compute per-dimension document vectors
- **Computational Costs**:
  - **BM25**: Fastest (CPU-only, no neural models)
  - **Dense/SPLADE**: Moderate (single forward pass per text)
  - **Late Interaction**: Higher (token-level interactions, O(|Q|×|D|×d) per pair)
  - **Cross-Encoder**: Highest (joint encoding, applied only to top-k candidates)
- **Optimization Tips**:
  - Use smaller `top_k` values to reduce reranking cost
  - Disable query planning for faster experimentation
  - Use smaller models for prototyping (e.g., `all-MiniLM-L6-v2` instead of `all-mpnet-base-v2`)
  - Enable `ablation.debug: false` for production runs (reduces logging overhead)

## Results and Findings

### Baseline Performance (Example Run)

Sample baseline results on small test dataset (`corpus-small.jsonl`, `queries-small.jsonl`):

```
Query query_0: recall@10=0.0,   nDCG@10=0.000
Query query_1: recall@10=0.5,   nDCG@10=0.185
Query query_2: recall@10=1.0,   nDCG@10=0.398
Query query_3: recall@10=0.0,   nDCG@10=0.000
Query query_4: recall@10=0.5,   nDCG@10=0.185
Query query_5: recall@10=0.0,   nDCG@10=0.000
Query query_6: recall@10=0.5,   nDCG@10=0.193
Query query_7: recall@10=1.0,   nDCG@10=0.500
Query query_8: recall@10=0.5,   nDCG@10=0.204
Query query_9: recall@10=0.0,   nDCG@10=0.000
Query query_10: recall@10=0.0,  nDCG@10=0.000
Query query_11: recall@10=0.0,  nDCG@10=0.000
Query query_12: recall@10=0.0,  nDCG@10=0.000
```

**Average:** Recall@10 ≈ 0.31, nDCG@10 ≈ 0.13

### Observations

- **High variance across queries**: Some queries achieve perfect recall (1.0), others retrieve no relevant documents (0.0)
- **Small test set limitations**: Results are on a minimal corpus for prototyping; production evaluation requires larger datasets (e.g., MS MARCO, BEIR)
- **Hybrid fusion impact**: Comparing individual retrievers (BM25-only, Dense-only, SPLADE-only) vs. hybrid fusion shows improved robustness

### Ablation Study Insights

Initial ablation experiments revealed:
- **Dimensionality impact**: Performance degradation observed when truncating embeddings below 256 dimensions
- **Diminishing returns**: Marginal improvements beyond 512 dimensions for this dataset
- **Model-specific patterns**: Dense retrievers more sensitive to dimensionality reduction than SPLADE

**Note:** Comprehensive ablation results are available in `runs/ablation_results.csv` after running `python run_ablation.py`.

### Recommendations for Production Use

1. **Dataset**: Replace `corpus-small.jsonl` with full-scale benchmarks (MS MARCO passages, BEIR tasks)
2. **Models**: Upgrade to larger models:
   - Dense: `sentence-transformers/msmarco-bert-base-dot-v5`
   - SPLADE: `naver/splade-cocondenser-ensembledistil` (already used)
   - Reranker: `cross-encoder/ms-marco-MiniLM-L-12-v2` or larger
3. **Query Planning**: Replace `tiny-gpt2` with instruction-tuned models (Llama-2-7b-chat, Mistral-7B-Instruct)
4. **Evaluation**: Test across multiple domains using BEIR benchmark suite
5. **Hyperparameter Tuning**: Grid search over fusion weights, normalization strategies, and top-k values

## Troubleshooting

### Common Issues

**1. CUDA out of memory**
- Reduce batch sizes in model encoding
- Use smaller models or CPU-only mode
- Process corpus in chunks

**2. Slow ablation experiments**
- Reduce `ablation.dims` to fewer dimensions
- Use smaller corpus/query sets for prototyping
- Disable `ablation.debug` to reduce logging overhead

**3. Poor retrieval performance**
- Check data format (JSONL with correct fields)
- Verify qrels contain relevance judgments for test queries
- Ensure models downloaded correctly (check Hugging Face cache)
- Try different fusion strategies/normalization methods

**4. Query planning errors**
- Check LLM model availability on Hugging Face
- Increase `max_new_tokens` if decompositions are truncated
- Disable query planning (`enabled: false`) to isolate issues

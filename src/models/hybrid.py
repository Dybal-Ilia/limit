import numpy as np
from src.models.base import BaseModel

# GOOGLE'S LIMIT PAPER UNDERSTANDING:
# This implements hybrid retrieval to test against Google's theoretical limitations.
# Google proves that for any embedding dimension d, there exist document combinations
# that cannot be retrieved by any query using single-vector embeddings.
# 
# THEORETICAL CONTEXT: The LIMIT dataset uses combinatorial structure where each query
# has exactly 2 relevant documents. This creates theoretical impossibility scenarios
# that break single-vector embeddings.
#
# OUR APPROACH: Hybrid retrieval combines BM25 (lexical), SPLADE (sparse semantic), 
# and dense embeddings to overcome these limitations through complementary strengths.
#
# WHY THIS MATTERS: Understanding these limitations is crucial for designing
# better retrieval systems and knowing when to use hybrid approaches.

class HybridRetriever(BaseModel):
    def __init__(self, retrievers, weights=None, fusion: str = "weighted", normalizer: str = "minmax", rrf_k: int = 60):
        self.retrievers = retrievers
        if weights is None:
            self.weights = [1.0] * len(retrievers)
        else:
            # pad or trim to length of retrievers to avoid silent dropping by zip
            if len(weights) < len(retrievers):
                self.weights = list(weights) + [1.0] * (len(retrievers) - len(weights))
            else:
                self.weights = list(weights)[:len(retrievers)]
        self.fusion = fusion  # 'weighted' | 'rrf'
        self.normalizer = normalizer  # 'minmax' | 'zscore' | 'rank'
        self.rrf_k = rrf_k

    def encode(self, texts):
        """
        NOTE: Hybrid retrieval uses score fusion in retrieve(), not embedding fusion.
        This method is only for interface compatibility - use retrieve() instead.
        
        CRITICAL: We cannot average embeddings from different retrievers because:
        1. BM25 doesn't produce embeddings - it produces relevance scores
        2. Different retrievers use different embedding spaces and scales
        3. Proper hybrid fusion requires combining SCORES, not embeddings
        """
        # Just use first retriever's embeddings for interface compatibility
        return self.retrievers[0].encode(texts)

    def retrieve(self, query, corpus, top_k):
        """
        Classical hybrid retrieval: combine BM25 (keyword) + SPLADE (sparse embedding) + Dense (dense embedding)
        Each retriever produces relevance scores, we normalize and fuse them.
        
        FUSION STRATEGY:
        1. Get results from each retriever with their scores
        2. Normalize scores per retriever to [0,1] range for fair combination
        3. Combine normalized scores using weighted sum
        4. Sort by fused scores and return top_k
        
        WHY NORMALIZATION MATTERS:
        - BM25: Produces keyword-based relevance scores (TF-IDF based)
        - SPLADE: Produces sparse embedding similarity scores (learned sparse representations)  
        - Dense: Produces dense embedding similarity scores (semantic similarity)
        - Different scales need normalization before fusion
        """
        # Get results from each retriever with their scores
        retriever_results = []
        for retriever, weight in zip(self.retrievers, self.weights):
            results = retriever.retrieve(query, corpus, top_k)
            retriever_results.append((retriever, weight, results))
        
        # Prepare score data structures for fusion
        prepared_results = []
        for retriever, weight, results in retriever_results:
            if not results:
                prepared_results.append((retriever, weight, []))
                continue

            # Extract scores
            scores = [float(r.get("score", 0.0)) for r in results]

            if self.fusion == "rrf":
                # For RRF, we operate on ranks directly; no normalization needed
                norm_results = []
                for rank, result in enumerate(sorted(results, key=lambda x: float(x.get("score", 0.0)), reverse=True), start=1):
                    # RRF contribution
                    rrf_score = 1.0 / (self.rrf_k + rank)
                    norm_results.append({"id": result.get("id"), "score": rrf_score})
                prepared_results.append((retriever, weight, norm_results))
                continue

            # Otherwise, normalize per retriever
            if self.normalizer == "minmax":
                min_score, max_score = (min(scores), max(scores)) if scores else (0.0, 0.0)
                if max_score > min_score:
                    normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]
                else:
                    normalized_scores = [1.0] * len(scores)
            elif self.normalizer == "zscore":
                import math
                mean = sum(scores) / len(scores) if scores else 0.0
                var = sum((s - mean) ** 2 for s in scores) / len(scores) if scores else 0.0
                std = math.sqrt(var)
                normalized_scores = [((s - mean) / std) if std > 1e-8 else 0.0 for s in scores]
                # Bring to [0,1] via sigmoid for stability
                normalized_scores = [1.0 / (1.0 + math.exp(-x)) for x in normalized_scores]
            elif self.normalizer == "rank":
                # Rank-based normalization: highest score gets 1.0, lowest gets 0.0
                order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                ranks = [0] * len(scores)
                for r, idx in enumerate(order):
                    ranks[idx] = r
                max_rank = max(ranks) if ranks else 1
                normalized_scores = [1.0 - (r / max_rank if max_rank > 0 else 0.0) for r in ranks]
            else:
                normalized_scores = scores  # fallback, should not happen

            norm_results = []
            for i, result in enumerate(results):
                norm_results.append({"id": result.get("id"), "score": normalized_scores[i] if i < len(normalized_scores) else 0.0})
            prepared_results.append((retriever, weight, norm_results))
        
        # Fuse normalized scores from all retrievers
        all_scores = {}
        for _retriever, weight, results in prepared_results:
            for result in results:
                doc_id = result.get("id")
                score_val = float(result.get("score", 0.0))
                if doc_id is None:
                    # skip if id missing; upstream retrievers should set ids
                    continue
                # Weighted combination
                all_scores[doc_id] = all_scores.get(doc_id, 0.0) + weight * score_val

        # Sort by fused scores and return top_k
        sorted_docs = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"id": doc_id, "score": score} for doc_id, score in sorted_docs]

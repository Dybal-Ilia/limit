from src.models.hybrid import HybridRetriever
from src.utils.logger import get_logger

# GOOGLE'S LIMIT PAPER UNDERSTANDING:
# This pipeline wires together candidate generation (hybrid fusion over multiple
# retrievers) and cross-encoder reranking. The hybrid stage combines lexical (BM25),
# sparse (SPLADE), and dense (bi-encoder) scores to mitigate single-vector
# limitations highlighted by the LIMIT paper; the reranker then refines precision.

logger = get_logger("pipeline")

class HybridRetrievalPipeline:
    def __init__(self, *retrievers, reranker, config: dict | None = None):
        # Read fusion settings from config if provided
        fusion_cfg = (config or {}).get("retriever", {})
        weights = fusion_cfg.get("weights")
        fusion = fusion_cfg.get("fusion", "weighted")
        normalizer = fusion_cfg.get("normalizer", "minmax")
        rrf_k = int(fusion_cfg.get("rrf_k", 60))

        self.hybrid = HybridRetriever(list(retrievers), weights=weights, fusion=fusion, normalizer=normalizer, rrf_k=rrf_k)
        self.reranker = reranker

    def run(self, query, corpus, top_k, rerank_k):
        initial = self.hybrid.retrieve(query, corpus, top_k)
        logger.info(f"Hybrid retrieved {len(initial)} candidates for query (top_k={top_k})")
        if not initial:
            return []

        returned_ids = set()
        for r in initial:
            rid = None
            if isinstance(r, dict):
                rid = r.get("id")
            else:
                # r might already be a scalar id/index
                rid = r
            # try to coerce numeric-like ids to int for matching
            try:
                if rid is not None:
                    if not isinstance(rid, (int, str)):
                        rid = int(rid)
                    else:
                        # if string containing digits, coerce to int
                        if isinstance(rid, str) and rid.isdigit():
                            rid = int(rid)
            except Exception:
                # leave rid as-is if coercion fails
                pass
            if rid is not None:
                returned_ids.add(rid)

        top_docs = []
        for idx, d in enumerate(corpus):
            did = d.get("id", idx) if isinstance(d, dict) else idx
            if did in returned_ids:
                top_docs.append(d)
        logger.info(f"Reranking {len(top_docs)} documents (rerank_k={rerank_k})")
        reranked = self.reranker.rerank(query, top_docs, rerank_k)
        logger.info(f"Reranker returned {len(reranked)} documents")
        return reranked

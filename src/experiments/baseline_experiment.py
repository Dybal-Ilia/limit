from src.retrieval.hybrid_retriever import HybridRetrievalPipeline
from src.evaluation.metrics import recall_at_k, ndcg_at_k
from src.utils.logger import get_logger
import csv
import os
import numpy as np
from src.query_planning.agent import QueryPlanningAgent

logger = get_logger("experiment")


class BaselineExperiment:
    def __init__(self, retrievers, reranker, corpus, queries, qrels, config: dict | None = None):
        self.config = config or {}
        self.pipeline = HybridRetrievalPipeline(*retrievers, reranker=reranker, config=self.config)
        self.corpus = corpus
        self.queries = queries
        # Normalize qrels to a mapping: qid -> list of relevant ids
        if isinstance(qrels, dict):
            self.qrels = qrels
        elif isinstance(qrels, list):
            # list of dicts expected like {'qid': <id>, 'relevant': [...]}
            mapping = {}
            for row in qrels:
                if isinstance(row, dict):
                    qid = row.get("qid") or row.get("_id") or row.get("id")
                    rel = row.get("relevant") or row.get("relevant_ids") or row.get("docs")
                    if qid is not None:
                        mapping[qid] = rel or []
            self.qrels = mapping
        else:
            self.qrels = {}

    def run(self):
        top_k = int(self.config.get("retriever", {}).get("top_k", 500))
        rerank_k = int(self.config.get("retriever", {}).get("rerank_k", 200))
        qp_cfg = self.config.get("query_planning", {})
        qp_enabled = bool(qp_cfg.get("enabled", False))
        qp_model = qp_cfg.get("model", "sshleifer/tiny-gpt2")
        qp_max_new = int(qp_cfg.get("max_new_tokens", 64))
        qp_merge = qp_cfg.get("merge", "union")  # 'union' or 'rrf'
        qp_rrf_k = int(qp_cfg.get("rrf_k", 60))

        agent = None
        if qp_enabled:
            try:
                agent = QueryPlanningAgent(model_name=qp_model, max_new_tokens=qp_max_new)
                logger.info("Query planning enabled with model=%s", qp_model)
            except Exception as e:
                logger.warning("Failed to initialize QueryPlanningAgent: %s; falling back to single-query mode", e)
                agent = None

        results = {}
        for idx, q in enumerate(self.queries):
            if (idx + 1) % 10 == 0:
                logger.info(f"Processing query {idx+1}/{len(self.queries)}")
            # query can have '_id' or 'id'
            qid = q.get("_id") if isinstance(q, dict) else None
            if qid is None and isinstance(q, dict):
                qid = q.get("id")
            text = q.get("text") if isinstance(q, dict) else str(q)
            logger.debug(f"Running pipeline for qid={qid}")
            if agent is None:
                preds = self.pipeline.run(text, self.corpus, top_k=top_k, rerank_k=rerank_k)
            else:
                # Decompose query into sub-queries, merge candidates, rerank once
                subqs = agent.decompose_query(text)
                logger.info("Decomposed into %d sub-queries", len(subqs))
                # Collect candidates per subquery
                candidate_scores = {}
                all_ids = set()
                for sidx, sq in enumerate(subqs):
                    sres = self.pipeline.hybrid.retrieve(sq, self.corpus, top_k)
                    if qp_merge == "rrf":
                        for rank, r in enumerate(sorted(sres, key=lambda x: x.get("score", 0.0), reverse=True), start=1):
                            did = r.get("id")
                            if did is None:
                                continue
                            score = 1.0 / (qp_rrf_k + rank)
                            candidate_scores[did] = candidate_scores.get(did, 0.0) + score
                            all_ids.add(did)
                    else:  # union
                        for r in sres:
                            did = r.get("id")
                            if did is None:
                                continue
                            # use max per subquery to avoid bias; track maximum score
                            candidate_scores[did] = max(candidate_scores.get(did, 0.0), float(r.get("score", 0.0)))
                            all_ids.add(did)
                # Build doc list for reranking
                top_ids = sorted(all_ids, key=lambda i: candidate_scores.get(i, 0.0), reverse=True)[:top_k]
                docs = []
                for idx2, d in enumerate(self.corpus):
                    did = d.get("id", idx2) if isinstance(d, dict) else idx2
                    if did in top_ids:
                        docs.append(d)
                preds = self.rerank_only(text, docs, rerank_k)

            # diagnostics: show retrieved ids and qrels
            pred_ids = [p.get("id") for p in preds]
            # canonicalize ids to strings for safe comparison with qrels
            pred_ids_str = [str(x) for x in pred_ids if x is not None]
            logger.info("Query %s: retrieved %d docs (sample ids=%s)", qid, len(pred_ids_str), pred_ids_str[:10])
            relevant = self.qrels.get(qid, [])
            logger.debug("qrels for %s (type=%s): %s", qid, type(relevant), str(relevant)[:200])

            # compute a simple intersection for debugging
            # coerce relevant ids to strings as well
            relevant_set = set()
            if isinstance(relevant, (list, tuple)):
                for r in relevant:
                    if isinstance(r, dict):
                        if r.get("id") is not None:
                            relevant_set.add(str(r.get("id")))
                    else:
                        relevant_set.add(str(r))
            elif isinstance(relevant, set):
                relevant_set = set(str(x) for x in relevant)

            intersect = set(pred_ids_str) & relevant_set
            logger.info("Query %s: intersection size=%d", qid, len(intersect))
            # pass original preds but compute metrics using canonicalized relevant ids
            results[qid] = {
                "recall@10": recall_at_k(preds, list(relevant_set), 10),
                "nDCG@10": ndcg_at_k(preds, list(relevant_set), 10)
            }
        return results

    def rerank_only(self, query_text: str, docs, rerank_k: int):
        return self.pipeline.reranker.rerank(query_text, docs, rerank_k)

    def run_ablation(self, config):
        """
        Run ablation over embedding dims specified in config['ablation']['dims'].
        Supports two modes: 'truncate' (vec[:d]) and 'pad' (zero-pad to d).
        Writes per-dim aggregated metrics to config['ablation']['output'] as CSV.
        """
        ab_cfg = config.get("ablation", {})
        enabled = ab_cfg.get("enabled", False)
        if not enabled:
            logger.info("Ablation disabled in config")
            return None

        mode = ab_cfg.get("mode", "truncate")
        dims = ab_cfg.get("dims", [32, 64, 128, 256, 512])
        out_path = ab_cfg.get("output", "runs/ablation_results.csv")
        debug = bool(ab_cfg.get("debug", False))

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        aggregated_rows = []

        # wrapper around pipeline to allow embedding dim modification
        for d in dims:
            logger.info("Running ablation for dim=%d mode=%s", d, mode)
            per_query_metrics = []
            per_query_rows = []

            # find retrievers in hybrid and monkeypatch encode once per-dimension
            retrievers = getattr(self.pipeline.hybrid, "retrievers", None)
            originals = []
            if retrievers:
                for r in retrievers:
                    orig_encode = getattr(r, "encode", None)
                    originals.append((r, orig_encode))
                    if orig_encode is None:
                        continue

                    # capture mode and d by value in the closure
                    def make_wrapped(orig_fn, mode_val, d_val):
                        def wrapped(texts, *a, **kw):
                            vecs = orig_fn(texts, *a, **kw)
                            if debug:
                                logger.info(f"Wrapped encode called: mode={mode_val}, d={d_val}, input_len={len(texts) if hasattr(texts, '__len__') else '?'}")
                            try:
                                import torch as _torch
                            except Exception:
                                _torch = None

                            # If original returned a torch Tensor, operate in torch to preserve type/device
                            if _torch is not None and isinstance(vecs, _torch.Tensor):
                                arr = vecs
                                if arr.dim() == 1:
                                    arr = arr.unsqueeze(0)
                                D = arr.size(1)
                                if mode_val == "truncate":
                                    if d_val < D:
                                        arr = arr[:, :d_val]
                                    elif d_val > D:
                                        pad = _torch.zeros((arr.size(0), d_val - D), dtype=arr.dtype, device=arr.device)
                                        arr = _torch.cat([arr, pad], dim=1)
                                elif mode_val == "pad":
                                    if d_val > D:
                                        pad = _torch.zeros((arr.size(0), d_val - D), dtype=arr.dtype, device=arr.device)
                                        arr = _torch.cat([arr, pad], dim=1)
                                    elif d_val < D:
                                        arr = arr[:, :d_val]
                                return arr

                            # Otherwise operate on numpy arrays/lists and return numpy
                            arr = np.asarray(vecs)
                            # vector shape (n, D)
                            if arr.ndim == 1:
                                arr = arr[np.newaxis, :]
                            D = arr.shape[1]
                            if mode_val == "truncate":
                                if d_val < D:
                                    arr = arr[:, :d_val]
                                elif d_val > D:
                                    pad = np.zeros((arr.shape[0], d_val - D), dtype=arr.dtype)
                                    arr = np.concatenate([arr, pad], axis=1)
                            elif mode_val == "pad":
                                if d_val > D:
                                    pad = np.zeros((arr.shape[0], d_val - D), dtype=arr.dtype)
                                    arr = np.concatenate([arr, pad], axis=1)
                                elif d_val < D:
                                    arr = arr[:, :d_val]
                            return arr
                        return wrapped

                    try:
                        wrapped = make_wrapped(orig_encode, mode, d)
                        try:
                            wrapped._ablation_info = (mode, d)
                        except Exception:
                            pass
                        setattr(r, "encode", wrapped)
                    except Exception as _e:
                        logger.debug("Failed to monkeypatch retriever %s: %s", getattr(r, '__class__', type(r)), _e)

                # optional debug: compare original vs wrapped outputs for the first retriever and first query (if available)
                if debug and self.queries:
                    sample_text = (self.queries[0].get("text") if isinstance(self.queries[0], dict) else str(self.queries[0]))
                    for r, orig in originals:
                        if orig is None:
                            continue
                        try:
                            sample_in = [sample_text]
                            orig_out = orig(sample_in)
                            wrapped_out = getattr(r, 'encode')(sample_in)
                            logger.info("Ablation debug retriever=%s mode=%s dim=%d orig_type=%s wrapped_type=%s", r.__class__.__name__, mode, d, type(orig_out), type(wrapped_out))
                            try:
                                import torch as _torch
                            except Exception:
                                _torch = None
                            def _shape(obj):
                                try:
                                    if _torch is not None and isinstance(obj, _torch.Tensor):
                                        return tuple(obj.size())
                                    import numpy as _np
                                    a = _np.asarray(obj)
                                    return a.shape
                                except Exception:
                                    return None
                            logger.info("Ablation debug shapes orig=%s wrapped=%s", _shape(orig_out), _shape(wrapped_out))
                        except Exception as _e:
                            logger.debug("Ablation debug sampling failed for retriever %s: %s", getattr(r, '__class__', type(r)), _e)

                # Precompute document encodings per-retriever for this dimension where possible.
                # We use the original encode() (stored in originals) to compute doc vectors,
                # then wrap the retriever.retrieve() method to use these precomputed doc vectors
                # (so both queries and documents are in the same d-dimensional space).
                texts = [d.get("text") if isinstance(d, dict) else str(d) for d in (self.corpus or [])]
                doc_vecs_map = {}
                # store original retrieve methods to restore later
                retrieve_originals = []
                try:
                    import torch as _torch
                except Exception:
                    _torch = None

                for r, orig_encode in originals:
                    orig_retrieve = getattr(r, "retrieve", None)
                    retrieve_originals.append((r, orig_retrieve))
                    if orig_encode is None:
                        # nothing to precompute for this retriever
                        continue
                    # compute doc vectors using the original encoder
                    try:
                        doc_vecs = orig_encode(texts)
                    except Exception as _e:
                        logger.debug("Failed to compute doc vectors for retriever %s: %s", getattr(r, '__class__', type(r)), _e)
                        continue

                    # utility to truncate/pad arrays/tensors
                    def _trunc_pad(vecs, mode_val, d_val):
                        # torch tensor
                        try:
                            if _torch is not None and isinstance(vecs, _torch.Tensor):
                                arr = vecs
                                if arr.dim() == 1:
                                    arr = arr.unsqueeze(0)
                                D = arr.size(1)
                                if mode_val == "truncate":
                                    if d_val < D:
                                        arr = arr[:, :d_val]
                                    elif d_val > D:
                                        pad = _torch.zeros((arr.size(0), d_val - D), dtype=arr.dtype, device=arr.device)
                                        arr = _torch.cat([arr, pad], dim=1)
                                elif mode_val == "pad":
                                    if d_val > D:
                                        pad = _torch.zeros((arr.size(0), d_val - D), dtype=arr.dtype, device=arr.device)
                                        arr = _torch.cat([arr, pad], dim=1)
                                    elif d_val < D:
                                        arr = arr[:, :d_val]
                                return arr
                        except Exception:
                            pass

                        # numpy fallback
                        arr = np.asarray(vecs)
                        if arr.ndim == 1:
                            arr = arr[np.newaxis, :]
                        D = arr.shape[1]
                        if mode_val == "truncate":
                            if d_val < D:
                                arr = arr[:, :d_val]
                            elif d_val > D:
                                pad = np.zeros((arr.shape[0], d_val - D), dtype=arr.dtype)
                                arr = np.concatenate([arr, pad], axis=1)
                        elif mode_val == "pad":
                            if d_val > D:
                                pad = np.zeros((arr.shape[0], d_val - D), dtype=arr.dtype)
                                arr = np.concatenate([arr, pad], axis=1)
                            elif d_val < D:
                                arr = arr[:, :d_val]
                        return arr

                    try:
                        doc_vecs = _trunc_pad(doc_vecs, mode, d)
                        doc_vecs_map[id(r)] = doc_vecs
                    except Exception as _e:
                        logger.debug("Failed to truncate/pad doc vecs for retriever %s: %s", getattr(r, '__class__', type(r)), _e)

                    # Wrap retrieve to use precomputed doc_vecs for known retriever types
                    def make_retrieve_with_docvecs(orig_retr_fn, retriever, doc_vecs_local, orig_enc):
                        def wrapped_retrieve(query, corpus_arg, top_k_arg):
                            cls_name = retriever.__class__.__name__
                            try:
                                # CRITICAL: Use the wrapped/monkeypatched encode() method, not original
                                # This ensures query vectors match document dimensionality (both truncated/padded)
                                qvec = retriever.encode([query]) if hasattr(retriever, 'encode') else None
                                
                                # Debug: log shapes if debug mode enabled
                                if debug and qvec is not None:
                                    try:
                                        q_shape = np.asarray(qvec).shape
                                        d_shape = np.asarray(doc_vecs_local).shape
                                        if hasattr(retriever, 'encode') and hasattr(retriever.encode, '_ablation_info'):
                                            mode_val, d_val = retriever.encode._ablation_info
                                            logger.info(f"Ablation {cls_name}: query_shape={q_shape}, doc_shape={d_shape}, target_dim={d_val}")
                                    except Exception:
                                        pass
                            except Exception:
                                qvec = None

                            # Dense retriever scoring
                            if cls_name == 'DenseRetriever' and qvec is not None:
                                try:
                                    q_arr = np.asarray(qvec)
                                    if q_arr.ndim == 2:
                                        q_arr = q_arr[0]
                                    doc_arr = np.asarray(doc_vecs_local)
                                    
                                    # Validate dimensions match
                                    if debug:
                                        logger.info(f"Dense: About to compute similarity - q_shape={q_arr.shape}, doc_shape={doc_arr.shape}")
                                    
                                    if q_arr.shape[0] != doc_arr.shape[1]:
                                        logger.warning(f"Dense: Dimension mismatch! q_dim={q_arr.shape[0]}, doc_dim={doc_arr.shape[1]} - falling back")
                                        return orig_retr_fn(query, corpus_arg, top_k_arg)
                                    
                                    sims = doc_arr.dot(q_arr)
                                    top_ids = sims.argsort()[::-1][:top_k_arg]
                                    results = []
                                    for i in top_ids:
                                        doc = corpus_arg[i]
                                        doc_id = doc.get('id', i) if isinstance(doc, dict) else i
                                        results.append({'id': doc_id, 'score': float(sims[i])})
                                    
                                    if debug and len(results) > 0:
                                        logger.info(f"Dense: Successfully computed {len(results)} results using {q_arr.shape[0]}-dim vectors")
                                    return results
                                except Exception as e:
                                    # fallback to original retrieve
                                    logger.warning(f"Dense: Exception in wrapped retrieve: {e} - falling back")
                                    return orig_retr_fn(query, corpus_arg, top_k_arg)

                            # SPLADE scoring (assume torch tensors)
                            if cls_name == 'SPLADERetriever' and qvec is not None:
                                try:
                                    # ensure torch available
                                    if _torch is not None:
                                        q_t = qvec
                                        if not isinstance(q_t, _torch.Tensor):
                                            q_t = _torch.tensor(np.asarray(q_t))
                                        doc_t = doc_vecs_local
                                        if not isinstance(doc_t, _torch.Tensor):
                                            doc_t = _torch.tensor(np.asarray(doc_t))
                                        
                                        # Validate dimensions
                                        if debug:
                                            logger.info(f"SPLADE: About to compute similarity - q_shape={q_t.shape}, doc_shape={doc_t.shape}")
                                        
                                        if q_t.shape[-1] != doc_t.shape[-1]:
                                            logger.warning(f"SPLADE: Dimension mismatch! q_dim={q_t.shape[-1]}, doc_dim={doc_t.shape[-1]} - falling back")
                                            return orig_retr_fn(query, corpus_arg, top_k_arg)
                                        
                                        # Flatten to 1D if needed
                                        if q_t.ndim > 1:
                                            q_t = q_t.squeeze(0)  # (1, dim) -> (dim,)
                                        if doc_t.ndim == 1:
                                            doc_t = doc_t.unsqueeze(0)  # (dim,) -> (1, dim) if single doc
                                        
                                        # Compute cosine similarity: (dim,) vs (N, dim)
                                        # Normalize vectors
                                        q_norm = q_t / (_torch.norm(q_t) + 1e-8)
                                        doc_norm = doc_t / (_torch.norm(doc_t, dim=1, keepdim=True) + 1e-8)
                                        # Dot product for cosine similarity
                                        sims = _torch.matmul(doc_norm, q_norm)
                                        sims = sims.detach().cpu().numpy()
                                        topk = min(top_k_arg, len(sims))
                                        top_idxs = np.argsort(sims)[::-1][:topk]
                                        results = []
                                        for idx in top_idxs:
                                            i = int(idx)
                                            doc = corpus_arg[i]
                                            doc_id = doc.get('id', i) if isinstance(doc, dict) else i
                                            results.append({'id': doc_id, 'score': float(sims[i])})
                                        
                                        if debug and len(results) > 0:
                                            logger.info(f"SPLADE: Successfully computed {len(results)} results using {q_t.shape[-1]}-dim vectors")
                                        return results
                                except Exception as e:
                                    logger.warning(f"SPLADE: Exception in wrapped retrieve: {e} - falling back")
                                    return orig_retr_fn(query, corpus_arg, top_k_arg)

                            # Default: fallback to original retrieve
                            return orig_retr_fn(query, corpus_arg, top_k_arg)
                        return wrapped_retrieve

                    # attach wrapper if original retrieve exists
                    if orig_retrieve is not None and id(r) in doc_vecs_map:
                        try:
                            new_retrieve = make_retrieve_with_docvecs(orig_retrieve, r, doc_vecs_map[id(r)], orig_encode)
                            setattr(r, 'retrieve', new_retrieve)
                        except Exception as _e:
                            logger.debug('Failed to wrap retrieve for retriever %s: %s', getattr(r, '__class__', type(r)), _e)

                # keep retrieve_originals for restoring at the end of this dimension
            else:
                retrieve_originals = []

            # run all queries with the retrievers patched for this dimension
            for idx, q in enumerate(self.queries):
                qid = q.get("_id") if isinstance(q, dict) else None
                if qid is None and isinstance(q, dict):
                    qid = q.get("id")
                text = q.get("text") if isinstance(q, dict) else str(q)

                preds = self.pipeline.run(text, self.corpus, top_k=int(self.config.get("retriever", {}).get("top_k", 500)), rerank_k=int(self.config.get("retriever", {}).get("rerank_k", 200)))

                relevant = self.qrels.get(qid, [])
                # canonicalize relevant
                rels = []
                if isinstance(relevant, (list, tuple)):
                    for r in relevant:
                        if isinstance(r, dict):
                            if r.get("id") is not None:
                                rels.append(str(r.get("id")))
                        else:
                            rels.append(str(r))

                recall2 = recall_at_k(preds, rels, 2)
                recall10 = recall_at_k(preds, rels, 10)
                recall100 = recall_at_k(preds, rels, 100)
                ndcg10 = ndcg_at_k(preds, rels, 10)

                per_query_metrics.append((qid, recall2, recall10, recall100, ndcg10))
                per_query_rows.append({
                    "qid": qid,
                    "recall@2": recall2,
                    "recall@10": recall10,
                    "recall@100": recall100,
                    "nDCG@10": ndcg10
                })

            # restore originals after finishing this dimension
            for r, orig in originals:
                try:
                    if orig is None:
                        if hasattr(r, 'encode'):
                            delattr(r, "encode")
                    else:
                        setattr(r, "encode", orig)
                except Exception:
                    pass
            # restore original retrieve methods if we wrapped them
            try:
                for r, orig_retr in retrieve_originals:
                    try:
                        if orig_retr is None:
                            # nothing to restore
                            continue
                        setattr(r, 'retrieve', orig_retr)
                    except Exception:
                        pass
            except Exception:
                # retrieve_originals may not be defined if no retrievers
                pass

            # aggregate means
            mean_rec2 = float(np.nanmean([m[1] for m in per_query_metrics]))
            mean_rec10 = float(np.nanmean([m[2] for m in per_query_metrics]))
            mean_rec100 = float(np.nanmean([m[3] for m in per_query_metrics]))
            mean_ndcg10 = float(np.nanmean([m[4] for m in per_query_metrics]))

            aggregated_rows.append({
                "dim": d,
                "mode": mode,
                "recall@2": mean_rec2,
                "recall@10": mean_rec10,
                "recall@100": mean_rec100,
                "nDCG@10": mean_ndcg10
            })

            # write intermediate results so long runs have partial output
            # write aggregated CSV
            with open(out_path, "w", newline='', encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=["dim", "mode", "recall@2", "recall@10", "recall@100", "nDCG@10"]) 
                writer.writeheader()
                for row in aggregated_rows:
                    writer.writerow(row)

            # write per-query CSV for this dim
            per_path = os.path.join(os.path.dirname(out_path), f"ablation_dim_{d}.csv")
            with open(per_path, "w", newline='', encoding="utf-8") as pf:
                pwriter = csv.DictWriter(pf, fieldnames=["qid", "recall@2", "recall@10", "recall@100", "nDCG@10"]) 
                pwriter.writeheader()
                for prow in per_query_rows:
                    pwriter.writerow(prow)

        logger.info("Ablation finished; results written to %s", out_path)
        return aggregated_rows

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# GOOGLE'S LIMIT PAPER UNDERSTANDING:
# Cross-encoder rerankers fundamentally differ from single-vector embeddings (bi-encoders).
# Instead of encoding query and document separately, they jointly encode the pair,
# allowing rich interaction between query and document tokens.
#
# THEORETICAL CONTEXT: While Google's LIMIT paper proves single-vector embeddings have
# limitations, cross-encoders bypass this by not using separate embeddings at all.
# They use the full transformer attention mechanism over concatenated query+document.
#
# WHY IT HELPS: Cross-encoders can capture fine-grained relevance signals that single
# vectors cannot represent. However, they're computationally expensive (O(n) forward
# passes for n documents), so we use two-stage retrieval: fast single-vector candidate
# generation, then precise cross-encoder reranking on top-k.
#
# PIPELINE ROLE: We retrieve top-k=500 candidates with hybrid fusion (BM25+SPLADE+Dense),
# then rerank the top rerank_k=200 with the cross-encoder for final precision. This
# combines computational efficiency with theoretical expressiveness.

class CrossEncoderReranker:
    def __init__(self, model_name, device: str | None = None, batch_size: int = 16):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def rerank(self, query, docs, top_k):
        # prepare texts
        texts = [d.get("text") if isinstance(d, dict) else str(d) for d in docs]
        if not texts:
            return []

        # Tokenize in a batched fashion and move tensors to device
        encodings = self.tokenizer([f"{query} [SEP] {t}" for t in texts], padding=True, truncation=True, return_tensors="pt")

        scores = [0.0] * len(texts)
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                end = min(start + self.batch_size, len(texts))
                batch_slice = slice(start, end)
                batch_inputs = {k: v[batch_slice].to(self.device) for k, v in encodings.items()}
                # use mixed precision on CUDA to speed up inference and reduce memory
                with torch.cuda.amp.autocast(enabled=(self.device != "cpu")):
                    out = self.model(**batch_inputs)
                logits = out.logits.squeeze(-1)
                if logits.dim() == 0:
                    batch_scores = [float(logits.item())]
                else:
                    batch_scores = logits.cpu().tolist()
                for offset, sc in enumerate(batch_scores):
                    scores[start + offset] = sc

        # sort by score only to avoid comparing doc objects
        reranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)[:top_k]
        results = []
        for s, d in reranked:
            doc_id = d.get("id") if isinstance(d, dict) else None
            results.append({"id": doc_id, "score": float(s)})
        return results

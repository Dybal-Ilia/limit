from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

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

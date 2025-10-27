from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from src.models.base import BaseModel

class SPLADERetriever(BaseModel):
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

    def encode(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs).logits
        return outputs.mean(dim=1)

    def retrieve(self, query, corpus, top_k):
        q_vec = self.encode([query])[0]
        texts = [d.get("text") if isinstance(d, dict) else str(d) for d in (corpus or [])]
        if not texts:
            return []
        doc_vecs = torch.stack([self.encode([t])[0] for t in texts])
        scores = torch.nn.functional.cosine_similarity(q_vec.unsqueeze(0), doc_vecs)
        topk = min(top_k, scores.shape[0])
        top_idxs = torch.topk(scores, k=topk).indices.tolist()
        results = []
        for idx in top_idxs:
            i = int(idx)
            doc = corpus[i]
            doc_id = doc.get("id", i) if isinstance(doc, dict) else i
            results.append({"id": doc_id, "score": float(scores[i])})
        return results

from typing import List
import torch
from transformers import AutoModel, AutoTokenizer
from src.models.base import BaseModel

# Simplified late interaction retriever (ColBERT-style):
# - Encodes query and documents into token-level vectors
# - Scores by max-sim over token pairs and sums
# NOTE: This is a minimal pedagogical implementation for experimentation.

class LateInteractionRetriever(BaseModel):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _encode_tokens(self, texts: List[str]):
        with torch.no_grad():
            enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            out = self.model(**enc)
            # Use last hidden states as token embeddings; L2 normalize
            reps = out.last_hidden_state
            reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps, enc.attention_mask

    def encode(self, texts):
        reps, _ = self._encode_tokens(texts)
        # Return CLS-pooled embedding for interface compatibility (not used by retrieve)
        return reps[:, 0, :].detach().cpu().numpy()

    def retrieve(self, query, corpus, top_k):
        if not corpus:
            return []
        q_reps, q_mask = self._encode_tokens([query])
        q_rep = q_reps[0]  # [Lq, D]
        q_att = q_mask[0].bool()

        texts = [d.get("text") if isinstance(d, dict) else str(d) for d in corpus]
        d_reps, d_mask = self._encode_tokens(texts)

        # Compute ColBERT-style score: sum over query tokens of max dot with doc tokens
        scores = []
        for i in range(d_reps.size(0)):
            dr = d_reps[i]  # [Ld, D]
            dm = d_mask[i].bool()
            # mask tokens
            q_valid = q_rep[q_att]
            d_valid = dr[dm]
            if q_valid.numel() == 0 or d_valid.numel() == 0:
                scores.append(0.0)
                continue
            # [Lq, Ld]
            sims = torch.matmul(q_valid, d_valid.transpose(0, 1))
            # max over doc tokens, sum over query tokens
            score = torch.max(sims, dim=1).values.sum().item()
            scores.append(score)

        # Top-k documents
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for i in top:
            doc = corpus[i]
            doc_id = doc.get("id", i) if isinstance(doc, dict) else i
            results.append({"id": doc_id, "score": float(scores[i])})
        return results

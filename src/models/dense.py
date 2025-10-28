from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from src.models.base import BaseModel

# GOOGLE'S LIMIT PAPER UNDERSTANDING:
# Dense retrievers use single-vector embeddings, which are exactly what Google's LIMIT
# paper proves have fundamental theoretical limitations. For any fixed dimension d,
# there exist document combinations that cannot be retrieved correctly.
#
# THEORETICAL CONTEXT: The LIMIT dataset's combinatorial structure (exactly 2 relevant
# docs per query) is designed to expose these limitations. Dense retrievers map both
# queries and documents to fixed-dimensional vectors and use cosine similarity.
#
# WHY IT FAILS: When multiple documents need to be retrieved for a single query, the
# single query vector cannot simultaneously be close to all relevant document vectors
# in the embedding space, especially when documents are semantically diverse.
#
# OUR MITIGATION: We combine dense retrieval with BM25 (lexical) and SPLADE (sparse
# learned) in a hybrid system to overcome these single-vector limitations through
# complementary signal sources.

class DenseRetriever(BaseModel):
    def __init__(self, model_name, device: str | None = None, batch_size: int = 32):
        # choose device automatically if not provided
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        # SentenceTransformer accepts a device argument
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts):
        # use SentenceTransformer's device-aware encode
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            device=self.device,
            normalize_embeddings=True,
        )

    def retrieve(self, query, corpus, top_k):
        texts = [d.get("text") if isinstance(d, dict) else str(d) for d in (corpus or [])]
        if not texts:
            return []

        q_vec = self.encode([query])[0]
        doc_vecs = self.encode(texts)

        # ensure numpy arrays
        q_vec = np.asarray(q_vec)
        doc_vecs = np.asarray(doc_vecs)
        sims = doc_vecs.dot(q_vec)
        top_ids = sims.argsort()[::-1][:top_k]
        results = []
        for i in top_ids:
            doc = corpus[i]
            doc_id = doc.get("id", i) if isinstance(doc, dict) else i
            results.append({"id": doc_id, "score": float(sims[i])})
        return results

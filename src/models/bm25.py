from rank_bm25 import BM25Okapi
from src.models.base import BaseModel

# GOOGLE'S LIMIT PAPER UNDERSTANDING:
# BM25 represents the classical lexical/keyword-based retrieval approach that Google's
# paper contrasts with embedding-based methods. While embeddings fail on LIMIT's
# combinatorial scenarios, BM25 provides complementary keyword-matching capabilities.
#
# THEORETICAL ROLE: BM25 excels at exact keyword matching but struggles with semantic
# similarity. In hybrid systems, it provides the lexical component that helps overcome
# embedding limitations through term-frequency and document-length normalization.
#
# LIMIT DATASET CONTEXT: The synthetic "who likes X" queries often contain direct
# keyword matches in documents, making BM25 a crucial component for the hybrid approach.

class BM25Retriever(BaseModel):
    def __init__(self, corpus):
        # store corpus and build tokenized corpus defensively
        self.corpus = corpus or []

        def _text_of(doc):
            if isinstance(doc, dict):
                return str(doc.get("text") or doc.get("content") or "")
            return str(doc)

        self.tokenized_corpus = [_text_of(doc).split() for doc in self.corpus]
        self.model = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None

    def encode(self, texts):
        return [text.split() for text in texts]

    def retrieve(self, query, corpus=None, top_k=100):
        # use provided corpus if given, otherwise use stored one
        use_corpus = corpus if corpus is not None else self.corpus
        if self.model is None:
            return []

        scores = self.model.get_scores(query.split())

        # sort by score only to avoid comparing doc objects
        top_docs = sorted(zip(scores, use_corpus), key=lambda x: x[0], reverse=True)[:top_k]

        results = []
        for idx, (s, doc) in enumerate(top_docs):
            # prefer explicit id, fall back to the document's index in the corpus
            if isinstance(doc, dict):
                doc_id = doc.get("id", None)
            else:
                doc_id = None
            # find the original index in use_corpus for fallback (safe and deterministic)
            try:
                orig_index = use_corpus.index(doc)
            except ValueError:
                orig_index = idx
            if doc_id is None:
                doc_id = orig_index
            results.append({"id": doc_id, "score": float(s)})
        return results

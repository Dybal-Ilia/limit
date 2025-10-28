import numpy as np

def recall_at_k(preds, relevant, k):
    retrieved = [str(p["id"]) for p in preds[:k] if p.get("id") is not None]
    if not relevant:
        return 0.0
    relevant_set = set(str(x) for x in relevant)
    rel = len(set(retrieved) & relevant_set)
    return rel / len(relevant_set)

def ndcg_at_k(preds, relevant, k):
    relevant_set = set(str(x) for x in relevant)
    dcg = sum([1 / np.log2(i+2) for i, p in enumerate(preds[:k]) if str(p.get("id")) in relevant_set])
    idcg = sum([1 / np.log2(i+2) for i in range(min(k, len(relevant_set)))])
    if idcg == 0:
        return 0.0
    return dcg / idcg

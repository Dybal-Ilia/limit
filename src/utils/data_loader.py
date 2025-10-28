import json
from src.utils.logger import get_logger

logger = get_logger("data_loader")

def load_jsonl(path):
    logger.info(f"Loading jsonl from {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def load_corpus_queries_qrels(config):
    corpus_path = config["paths"]["corpus"]
    queries_path = config["paths"]["queries"]
    qrels_path = config["paths"]["qrels"]
    corpus = load_jsonl(corpus_path)
    queries = load_jsonl(queries_path)
    qrels = load_jsonl(qrels_path)
    logger.info("Normalizing corpus documents (ensure id/text)")
    # normalize corpus: ensure each doc has an 'id' and 'text'
    normalized = []
    for idx, doc in enumerate(corpus):
        if isinstance(doc, dict):
            # respect common id keys like '_id' or 'id'
            doc_id = doc.get("id") if doc.get("id") is not None else doc.get("_id", idx)
            text = doc.get("text") or doc.get("content") or ""
            normalized.append({**doc, "id": doc_id, "text": text})
        else:
            normalized.append({"id": idx, "text": str(doc)})

    # Normalize qrels into a mapping: qid -> [doc_id, ...]
    logger.info("Normalizing qrels into mapping (qid -> list of doc ids)")
    qrels_mapping = {}
    if isinstance(qrels, dict):
        qrels_mapping = qrels
    elif isinstance(qrels, list):
        for row in qrels:
            if not isinstance(row, dict):
                continue
            # support common key names used in qrels files
            qid = row.get("query-id") or row.get("query_id") or row.get("qid") or row.get("_id") or row.get("query")
            doc_id = row.get("corpus-id") or row.get("corpus_id") or row.get("doc_id") or row.get("id") or row.get("corpus") or row.get("document")
            # some qrels store a list of relevant docs under a single row
            if qid is not None and isinstance(row.get("relevant"), (list, tuple)):
                qrels_mapping.setdefault(qid, []).extend(row.get("relevant") or [])
            elif qid is not None and doc_id is not None:
                qrels_mapping.setdefault(qid, []).append(doc_id)
    else:
        logger.warning("Unrecognized qrels format: %s", type(qrels))

    logger.info("Loaded %d corpus docs, %d queries, %d qrel entries", len(normalized), len(queries), len(qrels_mapping))
    return normalized, queries, qrels_mapping

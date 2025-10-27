#!/usr/bin/env python3
"""
Simple CLI to run ablation using project config.
Run with your environment, e.g.

uv run python3 run_ablation.py

This script loads config, data, instantiates retrievers and runs the ablation harness.
"""
import yaml
import os
import sys

try:
    from src.utils.data_loader import load_corpus_queries_qrels
    from src.models.bm25 import BM25Retriever
    from src.models.splade import SPLADERetriever
    from src.models.dense import DenseRetriever
    from src.models.reranker import CrossEncoderReranker
    from src.experiments.baseline_experiment import BaselineExperiment
except Exception as e:
    print("Import error:", e)
    print("Make sure you installed dependencies and added project root to PYTHONPATH. Example:")
    print("  source .venv/bin/activate")
    print("  pip install -e .")
    sys.exit(1)

CONFIG_PATH = os.path.join("src", "config", "config.yaml")
if not os.path.exists(CONFIG_PATH):
    print("Config file not found at", CONFIG_PATH)
    sys.exit(1)

with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)

corpus, queries, qrels = load_corpus_queries_qrels(cfg)

# instantiate retrievers
bm25 = BM25Retriever(corpus)
splade = SPLADERetriever(cfg['models']['splade'])
dense = DenseRetriever(cfg['models']['dense'])
rr = CrossEncoderReranker(cfg['models']['reranker'])

retrievers = [bm25, splade, dense]

exp = BaselineExperiment(retrievers, rr, corpus, queries, qrels, cfg)

print("Running ablation as configured in", CONFIG_PATH)
rows = exp.run_ablation(cfg)
print("Ablation returned:")
print(rows)

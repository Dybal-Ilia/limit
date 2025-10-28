import yaml
from src.utils.data_loader import load_corpus_queries_qrels
from src.utils.logger import get_logger
from src.models.bm25 import BM25Retriever
from src.models.splade import SPLADERetriever
from src.models.dense import DenseRetriever
from src.models.reranker import CrossEncoderReranker
from src.experiments.baseline_experiment import BaselineExperiment
from src.models.late_interaction import LateInteractionRetriever

logger = get_logger("main")

def main():
    logger.info("Loading configuration from src/config/config.yaml")
    with open("src/config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    logger.info("Loading corpus, queries and qrels")
    corpus, queries, qrels = load_corpus_queries_qrels(config)
    logger.info(f"Loaded corpus={len(corpus)} docs, queries={len(queries)} items")

    logger.info("Initializing retrievers and reranker")
    bm25 = BM25Retriever(corpus)
    splade = SPLADERetriever(config["models"]["splade"])
    dense = DenseRetriever(config["models"]["dense"])
    # Optional multi-vector retriever
    late_inter = None
    if config.get("models", {}).get("late_interaction"):
        try:
            late_inter = LateInteractionRetriever(config["models"]["late_interaction"])
            logger.info("Late interaction retriever initialized")
        except Exception as e:
            logger.warning("Failed to init late interaction retriever: %s", e)
    reranker = CrossEncoderReranker(config["models"]["reranker"])

    # Build retriever list (late interaction optional, appended for comparison use if desired)
    retrievers = [bm25, splade, dense]
    if late_inter is not None:
        retrievers.append(late_inter)

    experiment = BaselineExperiment(retrievers, reranker, corpus, queries, qrels, config)
    # If ablation is enabled in config, run ablation harness instead of normal experiment
    if config.get("ablation", {}).get("enabled", False):
        logger.info("Ablation enabled in config — running ablation harness")
        results = experiment.run_ablation(config)
    else:
        logger.info("Starting experiment run")
        results = experiment.run()

    if isinstance(results, dict):
        for qid, metrics in results.items():
            logger.info(f"Query {qid}: {metrics}")
    elif isinstance(results, list):
        # aggregated rows from ablation
        for row in results:
            logger.info(f"Ablation row: {row}")
    else:
        logger.info("No results returned by experiment")

if __name__ == "__main__":
    main()

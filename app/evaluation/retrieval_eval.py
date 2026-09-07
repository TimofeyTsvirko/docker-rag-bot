"""
Simple RAG retrieval evaluation: Precision@k and Recall@k.

Usage:
    python -m app.evaluation.retrieval_eval

Golden set format (JSON list):
[
  {
    "query": "How to mount a volume in Docker?",
    "relevant_sources": ["volumes.md", "docker-run.md"]
  },
  ...
]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from app.services.qdrant import similarity_search
from app.config import get_settings

logger = logging.getLogger(__name__)

GOLDEN_PATH = Path(__file__).parent / "golden_set.json"


def precision_at_k(retrieved_sources: List[str], relevant_sources: List[str], k: int) -> float:
    if k == 0:
        return 0.0
    top = retrieved_sources[:k]
    hits = sum(1 for s in top if s in relevant_sources)
    return hits / k


def recall_at_k(retrieved_sources: List[str], relevant_sources: List[str], k: int) -> float:
    if not relevant_sources:
        return 0.0
    top = set(retrieved_sources[:k])
    hits = sum(1 for s in relevant_sources if s in top)
    return hits / len(relevant_sources)


def evaluate_retrieval(golden: List[Dict[str, Any]], k: int = 5) -> Dict[str, float]:
    precisions = []
    recalls = []

    for item in golden:
        query = item["query"]
        relevant = item.get("relevant_sources", [])
        docs = similarity_search(query, k=k, score_threshold=0.0)
        retrieved = [d.metadata.get("source", "") for d in docs]

        p = precision_at_k(retrieved, relevant, k)
        r = recall_at_k(retrieved, relevant, k)
        precisions.append(p)
        recalls.append(r)
        logger.info("Q: %s | P@%d=%.2f R@%d=%.2f | retrieved=%s", query, k, p, k, r, retrieved)

    n = len(golden) or 1
    return {
        f"precision@{k}": sum(precisions) / n,
        f"recall@{k}": sum(recalls) / n,
        "num_queries": len(golden),
    }


def main():
    logging.basicConfig(level=logging.INFO)
    if not GOLDEN_PATH.exists():
        # create a minimal example golden set
        example = [
            {
                "query": "How to create a Docker volume?",
                "relevant_sources": ["volumes.md"],
            },
            {
                "query": "Что такое docker-compose?",
                "relevant_sources": ["compose.md"],
            },
            {
                "query": "How to limit memory of a container?",
                "relevant_sources": ["resources.md"],
            },
        ]
        GOLDEN_PATH.write_text(json.dumps(example, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Created example golden set at {GOLDEN_PATH}")

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    settings = get_settings()
    metrics = evaluate_retrieval(golden, k=settings.top_k)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

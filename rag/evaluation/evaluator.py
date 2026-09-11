"""
RAG Pipeline Evaluator — measures retrieval and generation quality.

Metrics:
- Retrieval relevance (keyword overlap with expected)
- Source document accuracy (did we retrieve from the right document?)
- Source page accuracy (did we retrieve from the right page?)
- Citation presence in generated answers

Usage:
    python -m rag.evaluation.evaluator
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rag.pipeline import RAGPipeline
from rag.schemas import RetrievalResult

logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).parent / "test_dataset.json"


def load_test_dataset(path: Path | None = None) -> list[dict]:
    """Load the evaluation test dataset."""
    dataset_path = path or DATASET_PATH
    with open(dataset_path, "r") as f:
        return json.load(f)


def evaluate_retrieval(
    pipeline: RAGPipeline,
    dataset: list[dict],
    top_k: int = 5,
) -> dict:
    """Evaluate retrieval quality against the test dataset.

    Args:
        pipeline: An initialised RAG pipeline.
        dataset: List of test cases with expected results.
        top_k: Number of chunks to retrieve per query.

    Returns:
        Dict with evaluation metrics.
    """
    results = []

    for i, test_case in enumerate(dataset):
        question = test_case["question"]
        expected_doc = test_case.get("expected_document", "")
        expected_page = test_case.get("expected_page", -1)
        expected_keywords = test_case.get("expected_keywords", [])

        logger.info("Evaluating [%d/%d]: %s", i + 1, len(dataset), question)

        try:
            retrieval_result = pipeline.retrieve(question, top_k=top_k)
        except Exception as exc:
            logger.error("Retrieval failed for '%s': %s", question, exc)
            results.append({
                "question": question,
                "status": "error",
                "error": str(exc),
            })
            continue

        # Evaluate metrics
        metrics = _evaluate_single(
            retrieval_result=retrieval_result,
            expected_doc=expected_doc,
            expected_page=expected_page,
            expected_keywords=expected_keywords,
        )
        metrics["question"] = question
        metrics["status"] = "ok"
        metrics["num_results"] = len(retrieval_result.chunks)
        results.append(metrics)

    # Aggregate
    successful = [r for r in results if r["status"] == "ok"]

    summary = {
        "total_queries": len(dataset),
        "successful_queries": len(successful),
        "failed_queries": len(results) - len(successful),
        "avg_keyword_recall": (
            sum(r["keyword_recall"] for r in successful) / len(successful)
            if successful else 0
        ),
        "document_hit_rate": (
            sum(1 for r in successful if r["document_match"]) / len(successful)
            if successful else 0
        ),
        "page_hit_rate": (
            sum(1 for r in successful if r["page_match"]) / len(successful)
            if successful else 0
        ),
        "details": results,
    }

    return summary


def _evaluate_single(
    retrieval_result: RetrievalResult,
    expected_doc: str,
    expected_page: int,
    expected_keywords: list[str],
) -> dict:
    """Evaluate a single retrieval result."""

    # Check document match
    retrieved_docs = {sc.chunk.document_name for sc in retrieval_result.chunks}
    document_match = any(expected_doc.lower() in d.lower() for d in retrieved_docs) if expected_doc else False

    # Check page match
    retrieved_pages = {sc.chunk.page_number for sc in retrieval_result.chunks}
    page_match = expected_page in retrieved_pages if expected_page > 0 else False

    # Check keyword recall
    all_content = " ".join(sc.chunk.content.lower() for sc in retrieval_result.chunks)
    if expected_keywords:
        keyword_hits = sum(1 for kw in expected_keywords if kw.lower() in all_content)
        keyword_recall = keyword_hits / len(expected_keywords)
    else:
        keyword_recall = 0.0

    # Top scores
    top_scores = [sc.score for sc in retrieval_result.chunks[:3]]

    return {
        "document_match": document_match,
        "page_match": page_match,
        "keyword_recall": round(keyword_recall, 3),
        "top_scores": top_scores,
        "retrieved_documents": list(retrieved_docs),
        "retrieved_pages": sorted(retrieved_pages),
    }


def print_evaluation_report(summary: dict) -> None:
    """Pretty-print the evaluation results."""
    print("\n" + "=" * 60)
    print("  RAG Pipeline Evaluation Report")
    print("=" * 60)
    print(f"  Total queries:      {summary['total_queries']}")
    print(f"  Successful:         {summary['successful_queries']}")
    print(f"  Failed:             {summary['failed_queries']}")
    print(f"  Avg keyword recall: {summary['avg_keyword_recall']:.1%}")
    print(f"  Document hit rate:  {summary['document_hit_rate']:.1%}")
    print(f"  Page hit rate:      {summary['page_hit_rate']:.1%}")
    print("=" * 60)

    for detail in summary["details"]:
        status = "✅" if detail["status"] == "ok" else "❌"
        print(f"\n  {status} {detail['question']}")
        if detail["status"] == "ok":
            print(f"     Keywords: {detail['keyword_recall']:.0%} | "
                  f"Doc: {'✓' if detail['document_match'] else '✗'} | "
                  f"Page: {'✓' if detail['page_match'] else '✗'}")
            if detail.get("top_scores"):
                print(f"     Top scores: {detail['top_scores']}")
        else:
            print(f"     Error: {detail.get('error', 'unknown')}")

    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    pipeline = RAGPipeline()
    dataset = load_test_dataset()

    print("Running RAG evaluation...")
    print("Note: Make sure you have ingested documents first.\n")

    summary = evaluate_retrieval(pipeline, dataset)
    print_evaluation_report(summary)

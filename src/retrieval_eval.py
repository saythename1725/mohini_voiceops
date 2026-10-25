"""Compare Gemini dense retrieval with Gemini+dense/BM25 RRF on gold evidence."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from .paths import APP_ROOT, ROOT, require_dataset_root
from .knowledge_base import KnowledgeBase


def _read(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _split(text):
    return [part.strip() for part in (text or "").split(";") if part.strip()]


def _matches(candidate, target):
    return candidate.casefold() == target.casefold() or candidate.rsplit(":", 1)[-1].casefold() == target.rsplit(":", 1)[-1].casefold()


def _evaluate_case(kb, case_id, case_type, question, annotations, top_k):
    gold = _split(annotations.get("gold_evidence", ""))
    hard_negatives = _split(annotations.get("hard_negatives", ""))
    output = {}
    for strategy in ("dense", "hybrid_rrf"):
        chunks = (kb.search(question, limit=top_k) if strategy == "dense"
                  else kb.search_hybrid(question, limit=top_k))
        ranked = [f"{chunk.source}:{chunk.section_id}" for chunk in chunks]
        ranks = [index for index, candidate in enumerate(ranked, 1)
                 if any(_matches(candidate, item) for item in gold)]
        relevant = [item for item in gold if any(_matches(candidate, item) for candidate in ranked)]
        negative_ranks = {negative: next((index for index, candidate in enumerate(ranked, 1)
                                          if _matches(candidate, negative)), None)
                          for negative in hard_negatives}
        output[strategy] = {
            "question_id": case_id,
            "case_type": case_type,
            "question": question,
            "strategy": strategy,
            "ranked_evidence": ranked,
            "gold_evidence": gold,
            "relevant_retrieved": relevant,
            "hard_negative_ranks": negative_ranks,
            "recall_at_k": len(relevant) / len(gold) if gold else None,
            "reciprocal_rank": 1.0 / min(ranks) if ranks else 0.0,
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/evaluation/retrieval")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--questions-only", action="store_true")
    args = parser.parse_args()
    require_dataset_root()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = APP_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    kb = KnowledgeBase(ROOT / "kb")
    cases = []
    for row in _read(ROOT / "practice" / "questions.csv"):
        cases.append((row["id"], "question", row["question"], row))
    if not args.questions_only:
        for row in _read(ROOT / "practice" / "conversations.csv"):
            cases.append((f"{row['conversation_id']}_T{int(row['turn']):02d}", "conversation",
                          row["customer_message"], row))
    all_results = {"dense": [], "hybrid_rrf": []}
    for index, (case_id, case_type, question, annotations) in enumerate(cases, start=1):
        print(f"[retrieval {index}/{len(cases)}] {case_id}", flush=True)
        case_results = _evaluate_case(kb, case_id, case_type, question, annotations, args.top_k)
        for strategy, result in case_results.items():
            all_results[strategy].append(result)
    metrics = {}
    for strategy, results in all_results.items():
        with (output_dir / f"retrieval_results_{strategy}.jsonl").open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        recall = [result["recall_at_k"] for result in results if result["recall_at_k"] is not None]
        metrics[strategy] = {
            "cases": len(results),
            "mean_recall_at_k": statistics.mean(recall) if recall else None,
            "mean_reciprocal_rank": statistics.mean(result["reciprocal_rank"] for result in results) if results else None,
            "gold_coverage_at_k": sum(result["reciprocal_rank"] > 0 for result in results) / len(results) if results else None,
        }
    metrics["top_k"] = args.top_k
    metrics["query_source"] = "Original question/customer_message; annotation fields were not included in retrieval queries."
    (output_dir / "retrieval_comparison.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

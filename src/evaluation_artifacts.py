"""Build deterministic, annotation-based artifacts from a practice_eval report."""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

from .paths import APP_ROOT, ROOT, require_dataset_root
from .data_store import DataStore
from .knowledge_base import KnowledgeBase
from .practice_eval import fact_check, forbidden_check

SCHEMA = [
    "question_id", "question", "expected_decision", "actual_decision",
    "expected_evidence", "retrieved_evidence", "answer", "answer_correct",
    "unsupported_claims", "expected_tools", "actual_tools", "tool_calls_correct",
    "latency_ms",
]


def _split(value):
    if isinstance(value, list):
        return [str(part).strip() for part in value if str(part).strip()]
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def _read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _acceptable(row):
    return {part.casefold() for part in _split(row.get("acceptable_decisions") or row.get("expected_decision"))}


def _language_ok(text, language):
    if language == "hi":
        return bool(re.search(r"[\u0900-\u097f]{3,}", text))
    if language == "hi-en":
        return bool(re.search(r"\b(hai|hain|ho|kya|kaise|kitne|kitna|mujhe|mera|meri|nahi|nahin|wapas|milega|aap)\b",
                              text.casefold()))
    return True


def _source_text(sources, kb, data_store):
    by_id = {chunk.section_id.casefold(): f"{chunk.title}\n{chunk.text}"
             for chunk in kb.chunks}
    texts = []
    for source in sources:
        source = str(source)
        section_id = source.rsplit(":", 1)[-1]
        if source.casefold().startswith("products.csv:"):
            product = data_store.product(section_id)
            if product:
                texts.append(json.dumps(product, ensure_ascii=False))
                continue
        texts.append(by_id.get(section_id.casefold(), ""))
    return "\n".join(texts)


def _tool_names(item):
    return [call.get("name", "") for call in item.get("tool_calls", [])]


def _unsupported_numeric_claims(answer, sources, item, kb, data_store):
    context = _source_text(sources, kb, data_store)
    context += "\n" + json.dumps(item.get("tool_results", []), ensure_ascii=False)
    context_folded = context.casefold().replace(",", "")
    numbers = re.findall(r"(?<![\w])\d[\d,]*(?:\.\d+)?%?(?![\w])", answer)
    unique = list(dict.fromkeys(numbers))
    return [f"Numeric/date value '{value}' is absent from retrieved evidence and tool results"
            for value in unique if value.replace(",", "").casefold() not in context_folded]


def _records(report, questions_path, conversations_path, kb, data_store):
    question_rows = {row["id"]: row for row in _read_csv(questions_path)}
    conversation_rows = {}
    for row in _read_csv(conversations_path):
        conversation_rows[(row["conversation_id"], int(row["turn"]))] = row
    cases = []
    diagnostics = []
    retrieval = []

    def add(item, row, case_id, question):
        answer = item.get("reply", "")
        sources = item.get("retrieved_evidence", item.get("sources", []))
        expected_tools_text = row.get("expected_tool_calls")
        expected_tools_annotated = expected_tools_text is not None
        expected_tools = re.findall(r"([a-z_]+)\s*\(", expected_tools_text or "")
        expected_tools = ["create_handover_ticket" if name == "create_ticket" else name
                          for name in expected_tools]
        actual_tools = _tool_names(item)
        must_include = row.get("must_include", "")
        must_not_include = row.get("must_not_include", "")
        _, missing = fact_check(answer, must_include)
        forbidden = forbidden_check(answer, must_not_include)
        unsupported = _unsupported_numeric_claims(answer, sources, item, kb, data_store)
        unsupported.extend(f"Dataset-forbidden claim: {phrase}" for phrase in forbidden)
        acceptable = _acceptable(row)
        decision_ok = not acceptable or item.get("decision", "answer").casefold() in acceptable
        answerable = str(row.get("answerable", "True")).casefold() != "false"
        uncertainty_markers = ("don't have", "do not have", "not available", "not sure", "no information",
                               "can't confirm", "cannot confirm", "pata nahi", "jaankari nahi", "जानकारी नहीं")
        answerability_ok = answerable or any(marker in answer.casefold() for marker in uncertainty_markers)
        lang_ok = _language_ok(answer, row.get("reply_language") or row.get("language", "en"))
        no_model_error = not item.get("llm_error")
        answer_correct = not missing and not forbidden and not unsupported and lang_ok and answerability_ok and no_model_error
        gold = _split(row.get("gold_evidence", ""))
        retrieved = list(sources)
        retrieved_keys = {source.casefold() for source in retrieved}
        retrieved_sections = {source.rsplit(":", 1)[-1].casefold() for source in retrieved}
        hits = [e for e in gold if e.casefold() in retrieved_keys or
                e.rsplit(":", 1)[-1].casefold() in retrieved_sections]
        first_rank = next((rank for rank, source in enumerate(retrieved, 1)
                           if source.casefold() in {e.casefold() for e in gold} or
                           source.rsplit(":", 1)[-1].casefold() in {e.rsplit(":", 1)[-1].casefold() for e in gold}), None)
        record = {
            "question_id": case_id,
            "question": question,
            "expected_decision": row.get("expected_decision", ""),
            "actual_decision": item.get("decision", "answer"),
            "expected_evidence": gold,
            "retrieved_evidence": retrieved,
            "answer": answer,
            "answer_correct": bool(answer_correct),
            "unsupported_claims": unsupported,
            "expected_tools": expected_tools,
            "actual_tools": actual_tools,
            "tool_calls_correct": bool(expected_tools_annotated and
                                        sorted(expected_tools) == sorted(actual_tools)),
            "latency_ms": item.get("latency_ms"),
        }
        cases.append(record)
        diagnostics.append({
            "question_id": case_id,
            "decision_correct": decision_ok,
            "missing_required_facts": missing,
            "forbidden_claims": forbidden,
            "unsupported_claims": unsupported,
            "language_correct": lang_ok,
            "answerability_check": answerability_ok,
            "model_error": item.get("llm_error"),
            "expected_tools_annotated": expected_tools_annotated,
            "expected_tool_expression": expected_tools_text,
            "reference_answer_for_audit_only": row.get("reference_answer"),
            "hard_negatives": _split(row.get("hard_negatives", "")),
            "must_include": _split(must_include),
            "must_not_include": _split(must_not_include),
            "entities": row.get("entities", ""),
            "language": row.get("language", ""),
            "reply_language": row.get("reply_language", row.get("language", "")),
            "answerable": row.get("answerable"),
            "intent": row.get("intent", row.get("scenario", "")),
            "missing_gold_evidence": [e for e in gold if e.casefold() not in retrieved_keys and
                                      e.rsplit(":", 1)[-1].casefold() not in retrieved_sections],
            "practice_evaluator_failures": item.get("failures", []),
        })
        retrieval.append({
            "question_id": case_id,
            "ranked_retrieved_evidence": retrieved,
            "gold_evidence": gold,
            "relevant_retrieved": hits,
            "recall_at_k": len(hits) / len(gold) if gold else None,
            "reciprocal_rank": 1.0 / first_rank if first_rank else 0.0,
        })

    for item in report.get("questions", []):
        question_id = item.get("id", "")
        row = question_rows.get(question_id, {})
        question_text = (item.get("input") if item.get("input_mode") == "asr_text"
                         else row.get("question", item.get("question", "")))
        add(item, row, question_id, question_text or "")
    for conversation in report.get("conversations", []):
        conversation_id = conversation.get("conversation_id", "")
        for item in conversation.get("turns", []):
            turn = int(item.get("turn", 0))
            row = conversation_rows.get((conversation_id, turn), {})
            add(item, row, f"{conversation_id}_T{turn:02d}", row.get("customer_message", ""))
    return cases, diagnostics, retrieval


def _summary(cases, diagnostics, retrieval):
    count = len(cases)
    decision_correct = [d["decision_correct"] for d in diagnostics]
    annotated_tools = [case for case, diagnostic in zip(cases, diagnostics)
                       if diagnostic["expected_tools_annotated"]]
    recalls = [item["recall_at_k"] for item in retrieval if item["recall_at_k"] is not None]
    latencies = [case["latency_ms"] for case in cases if isinstance(case["latency_ms"], (int, float))]
    return {
        "cases": count,
        "decision_accuracy": sum(decision_correct) / count if count else 0,
        "answer_correctness": sum(case["answer_correct"] for case in cases) / count if count else 0,
        "retrieval": {
            "cases_with_gold_evidence": len(recalls),
            "mean_recall_at_k": statistics.mean(recalls) if recalls else None,
            "mean_reciprocal_rank": statistics.mean(item["reciprocal_rank"] for item in retrieval) if retrieval else None,
        },
        "evidence_grounding_heuristic": sum(bool(case["unsupported_claims"]) is False for case in cases) / count if count else 0,
        "tool_correctness": {
            "annotated_cases": len(annotated_tools),
            "accuracy": sum(case["tool_calls_correct"] for case in annotated_tools) / len(annotated_tools)
            if annotated_tools else None,
        },
        "latency_ms": {
            "measured_cases": len(latencies),
            "mean": statistics.mean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "p95": sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)] if latencies else None,
        },
        "limitations": ["Unsupported-claim detection is a deterministic numeric/date check plus dataset forbidden phrases, not a semantic proof."],
    }


def _write_jsonl(path, rows, enforce_schema=False):
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            if enforce_schema and list(row) != SCHEMA:
                raise ValueError(f"Wrong evaluation schema for {row.get('question_id')}: {list(row)}")
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_one(report_path, output_dir, questions_path, conversations_path, kb,
               require_conversations=True):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cases, diagnostics, retrieval = _records(report, questions_path, conversations_path, kb,
                                             DataStore(ROOT / "data"))
    expected_question_ids = {row["id"] for row in _read_csv(questions_path)}
    actual_question_ids = {case["question_id"] for case in cases if "_T" not in case["question_id"]}
    if actual_question_ids != expected_question_ids:
        raise ValueError(f"Evaluation report is incomplete: questions missing={sorted(expected_question_ids - actual_question_ids)}, extra={sorted(actual_question_ids - expected_question_ids)}")
    if require_conversations:
        expected_turn_ids = {f"{row['conversation_id']}_T{int(row['turn']):02d}"
                             for row in _read_csv(conversations_path)}
        actual_turn_ids = {case["question_id"] for case in cases if "_T" in case["question_id"]}
        if actual_turn_ids != expected_turn_ids:
            raise ValueError(f"Evaluation report is incomplete: conversation turns missing={sorted(expected_turn_ids - actual_turn_ids)}, extra={sorted(actual_turn_ids - expected_turn_ids)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "evaluation_results.jsonl", cases, enforce_schema=True)
    _write_jsonl(output_dir / "conversation_evaluation_results.jsonl",
                 [case for case in cases if "_T" in case["question_id"]], enforce_schema=True)
    _write_jsonl(output_dir / "evaluation_diagnostics.jsonl", diagnostics)
    _write_jsonl(output_dir / "retrieval_results.jsonl", retrieval)
    summary = _summary(cases, diagnostics, retrieval)
    summary["retrieval_variant"] = report.get("retrieval_variant", "unknown")
    for group_name, predicate in (
        ("questions", lambda case: "_T" not in case["question_id"]),
        ("conversation_turns", lambda case: "_T" in case["question_id"]),
    ):
        indexes = [index for index, case in enumerate(cases) if predicate(case)]
        summary[group_name] = _summary([cases[index] for index in indexes],
                                       [diagnostics[index] for index in indexes],
                                       [retrieval[index] for index in indexes])
    (output_dir / "evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    failures = [case for case, diagnostic in zip(cases, diagnostics)
                if not case["answer_correct"] or not diagnostic["decision_correct"] or
                (diagnostic["expected_tools_annotated"] and not case["tool_calls_correct"])]
    with (output_dir / "failure_cases.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question_id", "question", "expected_decision", "actual_decision",
                                                   "answer", "unsupported_claims", "expected_evidence",
                                                   "retrieved_evidence", "suggested_fix"])
        writer.writeheader()
        for case in failures:
            diagnostic = next((row for row in diagnostics if row["question_id"] == case["question_id"]), {})
            reasons = diagnostic.get("practice_evaluator_failures", [])
            if case["unsupported_claims"]:
                fix = "Strengthen answer grounding and only state numbers supported by retrieved evidence or tool results."
            elif diagnostic.get("missing_gold_evidence"):
                fix = "Improve query coverage or section ranking for the missing gold evidence."
            elif not diagnostic.get("decision_correct", True):
                fix = "Review decision routing and the rules for answer, clarify, refuse, and escalate."
            elif not case["tool_calls_correct"]:
                fix = "Review tool selection, required arguments, and verification gates."
            elif diagnostic.get("missing_required_facts"):
                fix = "Ensure retrieved context includes the annotated policy facts and answer prompt preserves them."
            elif not diagnostic.get("language_correct", True):
                fix = "Improve response-language instruction and Hindi/Hinglish validation."
            else:
                fix = "Review this failure's diagnostic checks and relevant evidence before changing the agent."
            writer.writerow({**case,
                             "unsupported_claims": " | ".join(case["unsupported_claims"]),
                             "expected_evidence": " | ".join(case["expected_evidence"]),
                             "retrieved_evidence": " | ".join(case["retrieved_evidence"]),
                             "suggested_fix": fix + (" Failures: " + " | ".join(reasons) if reasons else "")})
    return cases, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Baseline practice_eval report JSON")
    parser.add_argument("--improved", help="Improved practice_eval report JSON")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--retrieval-comparison", help="Optional dense-vs-hybrid retrieval comparison JSON")
    parser.add_argument("--asr-report", help="Optional practice_eval report produced with --asr-text --questions-only")
    args = parser.parse_args()
    require_dataset_root()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = APP_ROOT / output_dir
    questions_path = ROOT / "practice" / "questions.csv"
    conversations_path = ROOT / "practice" / "conversations.csv"
    kb = KnowledgeBase(ROOT / "kb")
    baseline_cases, baseline = _write_one(Path(args.baseline), output_dir / "baseline",
                                          questions_path, conversations_path, kb)
    improved_cases, improved = (None, None)
    if args.improved:
        improved_cases, improved = _write_one(Path(args.improved), output_dir / "improved",
                                              questions_path, conversations_path, kb)
        _write_jsonl(output_dir / "evaluation_results.jsonl", improved_cases, enforce_schema=True)
        (output_dir / "evaluation_summary.json").write_text(json.dumps({"baseline": baseline, "improved": improved},
                                                                         ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "evaluation_diagnostics.jsonl").write_text(
            (output_dir / "improved" / "evaluation_diagnostics.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
        (output_dir / "retrieval_results.jsonl").write_text(
            (output_dir / "improved" / "retrieval_results.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
        (output_dir / "conversation_evaluation_results.jsonl").write_text(
            (output_dir / "improved" / "conversation_evaluation_results.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
        (output_dir / "failure_cases.csv").write_bytes((output_dir / "improved" / "failure_cases.csv").read_bytes())
    else:
        _write_jsonl(output_dir / "evaluation_results.jsonl", baseline_cases, enforce_schema=True)
        (output_dir / "evaluation_summary.json").write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
        for name in ("evaluation_diagnostics.jsonl", "retrieval_results.jsonl",
                     "conversation_evaluation_results.jsonl", "failure_cases.csv"):
            (output_dir / name).write_bytes((output_dir / "baseline" / name).read_bytes())
    retrieval_comparison = None
    if args.retrieval_comparison:
        retrieval_comparison = json.loads(Path(args.retrieval_comparison).read_text(encoding="utf-8"))
        summary_path = output_dir / "evaluation_summary.json"
        summary_json = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_json["retrieval_comparison"] = retrieval_comparison
        summary_path.write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")
    asr_summary = None
    if args.asr_report:
        _, asr_summary = _write_one(Path(args.asr_report), output_dir / "asr_text",
                                    questions_path, conversations_path, kb,
                                    require_conversations=False)
        (output_dir / "asr_text_evaluation_results.jsonl").write_text(
            (output_dir / "asr_text" / "evaluation_results.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
        summary_path = output_dir / "evaluation_summary.json"
        summary_json = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_json["asr_text"] = asr_summary
        summary_path.write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")
    report_lines = ["# Evaluation report", "", "All evaluation inputs came from the original practice CSV annotations. Expected answers and gold evidence were used only by this evaluator, never passed to the agent.", "",
                    "## Baseline", "", f"- Cases: {baseline['cases']}", f"- Decision accuracy: {baseline['decision_accuracy']:.1%}",
                    f"- Answer correctness: {baseline['answer_correctness']:.1%}",
                    f"- Mean Recall@k: {baseline['retrieval']['mean_recall_at_k']}",
                    f"- Mean MRR: {baseline['retrieval']['mean_reciprocal_rank']}",
                    f"- Tool accuracy on annotated cases: {baseline['tool_correctness']['accuracy']}",
                    f"- Measured latencies: {baseline['latency_ms']['measured_cases']}/{baseline['cases']}",
                    f"- Question decision accuracy: {baseline['questions']['decision_accuracy']:.1%}",
                    f"- Conversation-turn decision accuracy: {baseline['conversation_turns']['decision_accuracy']:.1%}", ""]
    if improved:
        report_lines += ["## Improved", "", f"- Cases: {improved['cases']}", f"- Decision accuracy: {improved['decision_accuracy']:.1%}",
                         f"- Answer correctness: {improved['answer_correctness']:.1%}",
                         f"- Mean Recall@k: {improved['retrieval']['mean_recall_at_k']}",
                         f"- Mean MRR: {improved['retrieval']['mean_reciprocal_rank']}",
                         f"- Tool accuracy on annotated cases: {improved['tool_correctness']['accuracy']}",
                         f"- Measured latencies: {improved['latency_ms']['measured_cases']}/{improved['cases']}",
                         f"- Question decision accuracy: {improved['questions']['decision_accuracy']:.1%}",
                         f"- Conversation-turn decision accuracy: {improved['conversation_turns']['decision_accuracy']:.1%}", ""]
        for metric_path, label in (("decision_accuracy", "Decision accuracy"),
                                   ("answer_correctness", "Answer correctness"),
                                   ("evidence_grounding_heuristic", "Evidence-grounding heuristic")):
            delta = improved[metric_path] - baseline[metric_path]
            report_lines.append(f"- {label} change: {delta:+.1%} (improved minus baseline)")
        for metric_path, label in (("mean_recall_at_k", "Mean Recall@k"),
                                   ("mean_reciprocal_rank", "Mean MRR")):
            before = baseline["retrieval"].get(metric_path)
            after = improved["retrieval"].get(metric_path)
            if before is not None and after is not None:
                report_lines.append(f"- {label} change: {after - before:+.3f} (improved minus baseline)")
        before_latency = baseline["latency_ms"].get("mean")
        after_latency = improved["latency_ms"].get("mean")
        if before_latency is not None and after_latency is not None:
            report_lines.append(f"- Mean latency change: {after_latency - before_latency:+.0f} ms (negative is faster)")
        before_tools = baseline["tool_correctness"].get("accuracy")
        after_tools = improved["tool_correctness"].get("accuracy")
        if before_tools is not None and after_tools is not None:
            report_lines.append(f"- Tool accuracy change: {after_tools - before_tools:+.1%}")
        report_lines.append("")
    if retrieval_comparison:
        dense = retrieval_comparison.get("dense", {})
        hybrid = retrieval_comparison.get("hybrid_rrf", {})
        report_lines += ["## Dense versus hybrid retrieval", "",
                         f"- Dense Recall@{retrieval_comparison.get('top_k')}: {dense.get('mean_recall_at_k')}",
                         f"- Hybrid RRF Recall@{retrieval_comparison.get('top_k')}: {hybrid.get('mean_recall_at_k')}",
                         f"- Dense MRR: {dense.get('mean_reciprocal_rank')}",
                         f"- Hybrid RRF MRR: {hybrid.get('mean_reciprocal_rank')}", ""]
        recall_better = (hybrid.get("mean_recall_at_k") or 0) >= (dense.get("mean_recall_at_k") or 0)
        mrr_better = (hybrid.get("mean_reciprocal_rank") or 0) >= (dense.get("mean_reciprocal_rank") or 0)
        strict_gain = (hybrid.get("mean_recall_at_k") != dense.get("mean_recall_at_k") or
                       hybrid.get("mean_reciprocal_rank") != dense.get("mean_reciprocal_rank"))
        report_lines.append("- Retrieval choice: " +
                            ("hybrid RRF is eligible for the full improved run because both retrieval metrics are non-regressing and at least one improves."
                             if recall_better and mrr_better and strict_gain else
                             "dense-only remains selected because the hybrid result did not improve both retrieval metrics without regression."))
        report_lines.append("")
    if asr_summary:
        report_lines += ["## ASR transcript-text variant", "",
                         f"- Cases: {asr_summary['cases']}",
                         f"- Decision accuracy: {asr_summary['decision_accuracy']:.1%}",
                         f"- Answer correctness: {asr_summary['answer_correctness']:.1%}",
                         "- This evaluates behavior on supplied ASR-like text, not the audio recognition model.", ""]
    report_lines += ["## Changes included in the candidate", "",
                     "- Gemini retrieval is strict: provider failures are surfaced and never switch to another retrieval method; transient Gemini rate/network errors receive bounded retries.",
                     "- Retrieval traces are retained even when response-language validation replaces a bad-language answer.",
                     "- Query-sensitive section coverage and prompt rules address refund-date calculation, warranty evidence, account-fraud handover, underspecified return windows, and unsupported-topic refusals.",
                     "- No canned policy answers were added; the LLM still answers from retrieved evidence and structured tool results.",
                     "- Local Ollama continues to handle query rewriting and response generation; Gemini is used for embeddings.", "",
                     "## Architecture decisions", "", "- Gemini embeddings are mandatory for agent retrieval; Gemini errors are reported and never switch to keyword retrieval.",
                     "- Dense-only and Gemini+dense/BM25 RRF are separate evaluation variants. The production agent stays on Gemini dense unless measured evidence supports adopting RRF.",
                     "- No reranker was added; there is no measured reranker result to justify an extra model call.",
                     "- LangGraph was not adopted. The current agent has a compact, explicit tool loop and per-conversation state; graph orchestration would add a dependency without a measured routing or state benefit.", "",
                     "## Limitations", "", "- Unsupported-claim checks use numbers and dates plus explicit dataset prohibitions; they cannot detect every semantic hallucination.",
                     "- The supplied question set has no expected-tool-call field, so tool correctness is measured only on annotated conversation turns.",
                     "- The standalone dense-versus-hybrid retrieval experiment uses each original customer query, before the agent's LLM query rewrite, to compare the retrievers on identical inputs.",
                     "- Voice is not represented by an annotated audio test set. Speech-like `asr_text` variants can be run separately, but do not measure ASR or TTS quality.",
                     "- This script does not alter the original dataset.", ""]
    (output_dir / "evaluation_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    main()

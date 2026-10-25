"""Detailed, per-case evaluation against the supplied capstone practice data."""

import argparse
import csv
import json
from pathlib import Path
import re
import sys
import time

from .agent import ROOT, SupportAgent


def _facts(field: str) -> list[list[str]]:
    return [[alternative.strip().casefold() for alternative in fact.split(" / ") if alternative.strip()]
            for fact in field.split(" ; ") if fact.strip()]


def fact_check(text: str, field: str) -> tuple[bool, list[str]]:
    folded = text.casefold()
    missing = [" / ".join(options) for options in _facts(field)
               if not any(option in folded for option in options)]
    return not missing, missing


def forbidden_check(text: str, field: str) -> list[str]:
    folded = text.casefold()
    return [phrase.strip() for phrase in field.split(" ; ")
            if phrase.strip() and phrase.strip().casefold() in folded]


def evidence_check(sources: list[str], field: str) -> list[str]:
    expected = [part.strip() for part in field.split(" ; ") if part.strip()]
    normalized = {source.casefold() for source in sources}
    found_sections = {source.rsplit(":", 1)[-1].casefold() for source in sources}
    return [item for item in expected
            if item.casefold() not in normalized and item.rsplit(":", 1)[-1].casefold() not in found_sections]


def expected_tools(field: str) -> list[str]:
    names = re.findall(r"([a-z_]+)\s*\(", field or "")
    return ["create_handover_ticket" if name == "create_ticket" else name for name in names]


def _save_progress(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    _save_progress_csv(path.with_name(f"{path.stem}.csv"), state)


def _save_progress_csv(path: Path, state: dict) -> None:
    """Write a flat, spreadsheet-friendly view alongside the resume checkpoint."""
    columns = ("case_type", "case_id", "turn", "passed", "language", "decision", "input", "reply",
               "failures", "missing_facts", "forbidden_claims", "missing_evidence",
               "expected_tools", "actual_tools", "sources", "latency_ms", "llm_used", "llm_error")
    rows = []

    def analysis_row(case_type: str, case_id: str, turn: str, item: dict, input_text: str) -> dict:
        checks = item.get("checks", {})
        flat = {
            "case_type": case_type, "case_id": case_id, "turn": turn,
            "passed": item.get("passed", False), "language": item.get("language", ""),
            "decision": item.get("decision", ""), "input": input_text,
            "reply": item.get("reply", ""), "llm_used": item.get("llm_used", False),
            "llm_error": item.get("llm_error") or "", "latency_ms": item.get("latency_ms"),
        }
        for column, values in (
            ("failures", item.get("failures", [])),
            ("missing_facts", checks.get("missing_facts", [])),
            ("forbidden_claims", checks.get("forbidden_claims", [])),
            ("missing_evidence", checks.get("missing_evidence", [])),
            ("expected_tools", checks.get("expected_tools", [])),
            ("actual_tools", checks.get("actual_tools", [])),
            ("sources", item.get("retrieved_evidence", item.get("sources", []))),
        ):
            flat[column] = " | ".join(map(str, values))
        return flat

    for question_id, item in state.get("questions", {}).items():
        rows.append(analysis_row("question", question_id, "", item,
                                 item.get("input", item.get("question", ""))))
    for conversation_id, turns in state.get("conversations", {}).items():
        for turn_id, item in turns.items():
            rows.append(analysis_row("conversation", conversation_id, turn_id, item,
                                     item.get("customer_message", "")))

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _grade(row: dict, result: dict) -> dict:
    reply = result["reply"]
    language = (row.get("language") or "").casefold()
    if language == "hi":
        language_ok = bool(re.search(r"[\u0900-\u097f]{3,}", reply))
    elif language == "hi-en":
        language_ok = bool(set(re.findall(r"[a-z]+", reply.casefold())) &
                           {"hai", "hain", "ho", "kya", "kaise", "kitne", "kitna", "mujhe", "mera",
                            "meri", "pe", "par", "nahi", "nahin", "kar", "kare", "wapas", "milega", "aap"})
    else:
        language_ok = True
    facts_ok, missing_facts = fact_check(reply, row.get("must_include", ""))
    forbidden = forbidden_check(reply, row.get("must_not_include", ""))
    acceptable = [part.strip().casefold() for part in
                  (row.get("acceptable_decisions") or row.get("expected_decision", "")).split(";") if part.strip()]
    decision_ok = not acceptable or result.get("decision", "answer").casefold() in acceptable
    missing_evidence = evidence_check(result.get("retrieved_evidence", result.get("sources", [])),
                                      row.get("gold_evidence", ""))
    expected = expected_tools(row.get("expected_tool_calls", ""))
    actual = [call.get("name", "") for call in result.get("tool_calls", [])]
    missing_tools = [name for name in expected if name not in actual]
    failures = []
    if missing_facts:
        failures.append("missing required fact(s): " + ", ".join(missing_facts))
    if forbidden:
        failures.append("forbidden claim(s): " + ", ".join(forbidden))
    if not decision_ok:
        failures.append(f"decision expected {acceptable}, got {result.get('decision', 'answer')}")
    if missing_evidence:
        failures.append("missing gold evidence: " + ", ".join(missing_evidence))
    if missing_tools:
        failures.append("missing expected tool call(s): " + ", ".join(missing_tools))
    if not language_ok:
        failures.append(f"reply language does not match expected language '{language}'")
    if result.get("llm_error"):
        failures.append("LLM unavailable: " + result["llm_error"])
    return {"passed": not failures, "failures": failures,
            "checks": {"required_facts_passed": facts_ok, "missing_facts": missing_facts,
                       "forbidden_claims": forbidden, "decision_passed": decision_ok,
                       "evidence_passed": not missing_evidence, "missing_evidence": missing_evidence,
                       "expected_tools": expected, "actual_tools": actual,
                       "tool_calls_passed": not missing_tools,
                       "language_passed": language_ok},
            "decision": result.get("decision", "answer"), "reply": reply,
            "sources": result.get("sources", []), "tool_calls": result.get("tool_calls", []),
            "retrieved_evidence": result.get("retrieved_evidence", result.get("sources", [])),
            "tool_results": result.get("tool_results", []),
            "llm_used": result.get("llm_used", False), "llm_error": result.get("llm_error")}


def evaluate_questions(agent: SupportAgent, path: Path, progress: bool = True,
                       question_id: str | None = None, use_asr_text: bool = False,
                       cached: dict | None = None, on_result=None) -> list[dict]:
    results = []
    cached = cached if cached is not None else {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if question_id and row["id"].casefold() != question_id.casefold():
                continue
            if row["id"] in cached:
                results.append(cached[row["id"]])
                if progress:
                    print(f"[question {row['id']}] already saved; skipping", flush=True)
                continue
            if progress:
                print(f"[question {row['id']}] running ({row.get('language', 'unknown')})...", flush=True)
            agent.state = agent.state.__class__()
            customer_input = (row.get("asr_text") or row["question"]) if use_asr_text else row["question"]
            started = time.perf_counter()
            result = agent.respond(customer_input)
            latency_ms = round((time.perf_counter() - started) * 1000)
            graded = _grade(row, result)
            graded.update({"id": row["id"], "language": row.get("language"),
                           "question": row["question"], "input": customer_input,
                           "input_mode": "asr_text" if use_asr_text else "question",
                           "latency_ms": latency_ms,
                           "expected_decision": row.get("expected_decision")})
            results.append(graded)
            cached[row["id"]] = graded
            if on_result:
                on_result()
            if progress:
                status = "PASS" if graded["passed"] else "FAIL: " + " | ".join(graded["failures"])
                print(f"[question {row['id']}] {status}", flush=True)
    return results


def evaluate_conversations(agent_factory, path: Path, progress: bool = True,
                           cached: dict | None = None, on_result=None) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            groups.setdefault(row["conversation_id"], []).append(row)
    results = []
    cached = cached if cached is not None else {}
    for conversation_id, turns in groups.items():
        ordered_turns = sorted(turns, key=lambda item: int(item["turn"]))
        saved_turns = cached.get(conversation_id, {})
        expected_turn_ids = {int(row["turn"]) for row in ordered_turns}
        if set(map(int, saved_turns)) == expected_turn_ids:
            restored = [saved_turns[str(turn_id)] for turn_id in sorted(expected_turn_ids)]
            results.append({"conversation_id": conversation_id,
                            "passed": all(item["passed"] for item in restored), "turns": restored})
            if progress:
                print(f"[conversation {conversation_id}] already saved; skipping", flush=True)
            continue
        # Partial conversations are rerun from turn one to reconstruct in-memory state safely.
        saved_turns = {}
        cached[conversation_id] = saved_turns
        agent = agent_factory()
        turn_results = []
        for row in ordered_turns:
            if progress:
                print(f"[conversation {conversation_id} turn {row['turn']}] running ({row.get('language', 'unknown')})...", flush=True)
            started = time.perf_counter()
            result = agent.respond(row["customer_message"])
            latency_ms = round((time.perf_counter() - started) * 1000)
            graded = _grade(row, result)
            graded.update({"turn": int(row["turn"]), "language": row.get("language"),
                           "customer_message": row["customer_message"],
                           "latency_ms": latency_ms,
                           "expected_decision": row.get("expected_decision")})
            turn_results.append(graded)
            saved_turns[str(row["turn"])] = graded
            if on_result:
                on_result()
            if progress:
                status = "PASS" if graded["passed"] else "FAIL: " + " | ".join(graded["failures"])
                print(f"[conversation {conversation_id} turn {row['turn']}] {status}", flush=True)
        results.append({"conversation_id": conversation_id,
                        "passed": all(item["passed"] for item in turn_results), "turns": turn_results})
    return results


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Evaluate against all supplied ZentroMart practice cases.")
    parser.add_argument("--output", help="Report file (defaults to outputs/practice_report.json)")
    parser.add_argument("--quiet", action="store_true", help="Hide per-case progress and failure logs")
    parser.add_argument("--question-id", help="Run one question only, for example PQ0002")
    parser.add_argument("--questions-only", action="store_true", help="Skip the conversation set")
    parser.add_argument("--asr-text", action="store_true", help="Evaluate speech-like asr_text variants instead of clean questions")
    parser.add_argument("--resume", action="store_true", help="Resume an incomplete/saved run from outputs/practice_progress.json")
    parser.add_argument("--fresh", action="store_true", help="Ignore saved progress and start a new evaluation")
    parser.add_argument("--checkpoint", help="Checkpoint JSON path (defaults to outputs/practice_progress.json)")
    parser.add_argument("--retrieval", choices=("dense", "hybrid"), default="dense",
                        help="Evaluation variant; hybrid combines Gemini dense rankings with BM25 via RRF.")
    args = parser.parse_args()
    config = {"question_id": args.question_id, "asr_text": args.asr_text,
              "questions_only": args.questions_only, "retrieval": args.retrieval}
    checkpoint = Path(args.checkpoint or "outputs/practice_progress.json")
    if not checkpoint.is_absolute():
        checkpoint = Path(__file__).resolve().parents[1] / checkpoint
    saved = None
    if args.resume or (not args.fresh and checkpoint.exists()):
        try:
            candidate = json.loads(checkpoint.read_text(encoding="utf-8"))
            if candidate.get("config") == config and not candidate.get("complete", False):
                saved = candidate
            elif args.resume:
                parser.error("No matching incomplete progress file exists. Start a new run or check --question-id/--asr-text options.")
        except (OSError, json.JSONDecodeError):
            if args.resume:
                parser.error("Could not read the saved progress file; use --fresh to start over.")
    if saved is None:
        saved = {"config": config, "complete": False, "questions": {}, "conversations": {}}
    elif not args.quiet:
        print(f"[resume] loaded {len(saved['questions'])} questions and "
              f"{sum(len(turns) for turns in saved['conversations'].values())} conversation turns", flush=True)
    persist = lambda: _save_progress(checkpoint, saved)
    persist()
    def factory():
        agent = SupportAgent(use_llm=True, verbose=not args.quiet)
        if args.retrieval == "hybrid":
            agent.kb.search = agent.kb.search_hybrid
        return agent
    progress = not args.quiet
    questions = evaluate_questions(factory(), ROOT / "practice" / "questions.csv", progress,
                                   args.question_id, args.asr_text, saved["questions"], persist)
    conversations = [] if (args.questions_only or args.question_id) else evaluate_conversations(
        factory, ROOT / "practice" / "conversations.csv", progress, saved["conversations"], persist)
    passed = sum(item["passed"] for item in questions)
    accuracy = (100.0 * passed / len(questions)) if questions else 0.0
    report = {"mode": "final_rag_tools", "retrieval_variant": args.retrieval,
              "questions_passed": passed, "questions_total": len(questions),
              "accuracy_percent": round(accuracy, 2), "questions": questions,
              "conversations": conversations}
    output = Path(args.output or "outputs/practice_report.json")
    if not output.is_absolute():
        output = Path(__file__).resolve().parents[1] / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    saved["complete"] = True
    persist()
    q_pass = sum(item["passed"] for item in questions)
    c_pass = sum(item["passed"] for item in conversations)
    turn_pass = sum(turn["passed"] for item in conversations for turn in item["turns"])
    turn_count = sum(len(item["turns"]) for item in conversations)
    print(f"Questions: {q_pass}/{len(questions)} passed")
    print(f"Question accuracy: {accuracy:.2f}% ({passed}/{len(questions)})")
    print(f"Conversation turns: {turn_pass}/{turn_count} passed")
    print(f"Entire conversations passed: {c_pass}/{len(conversations)}")
    print(f"Detailed report: {output}")
    print(f"Progress checkpoint: {checkpoint}")
    print(f"Flat analysis CSV: {checkpoint.with_name(f'{checkpoint.stem}.csv')}")


if __name__ == "__main__":
    main()

"""Lightweight practice-set evaluation for the initial implementation."""

import csv
from pathlib import Path

from .agent import SupportAgent, ROOT


def evaluate_questions(path: str | Path = ROOT / "practice" / "questions.csv", use_llm: bool = True) -> dict:
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8", newline="")))
    agent = SupportAgent(use_llm=use_llm)
    passed = 0
    for row in rows:
        agent.state = agent.state.__class__()
        result = agent.respond(row["question"])
        text = result["reply"].lower()
        required = [fact.strip().lower() for fact in row["must_include"].split(" ; ") if fact.strip()]
        if all(any(option.strip().lower() in text for option in fact.split(" / ")) for fact in required):
            passed += 1
    return {"passed": passed, "total": len(rows), "accuracy": passed / len(rows) if rows else 0.0}


if __name__ == "__main__":
    print(evaluate_questions())

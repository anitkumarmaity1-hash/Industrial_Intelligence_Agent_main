"""
Print app.rag.evaluation's RELEVANCE_CASES report as Markdown
(production-readiness fix 20).

tests/test_rag.py already hard-asserts on this report for the local
(BM25) backend on every CI run — that assertion is the actual regression
gate, not this script. What was missing (the audit's "RAG evaluation
metrics... run as part of CI, not ad hoc" item) was making the numbers
themselves visible without opening test output: this prints hit rate,
strict-pass rate and MRR as a small Markdown table, meant to be appended
to $GITHUB_STEP_SUMMARY (see .github/workflows/tests.yml) so a quiet
regression — still passing the hard threshold, but trending down — shows
up in the job summary every single run, not just when it finally crosses
the line that fails the build.

Usage:
    python scripts/report_rag_evaluation.py                  # prints to stdout
    python scripts/report_rag_evaluation.py >> "$GITHUB_STEP_SUMMARY"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.evaluation import RELEVANCE_CASES, evaluate_retriever  # noqa: E402
from app.rag.retriever import get_retriever  # noqa: E402


def main() -> None:
    retriever = get_retriever()
    report = evaluate_retriever(retriever, RELEVANCE_CASES)

    print("### RAG retrieval evaluation")
    print()
    print(f"Backend: `{report.backend}` · {len(report.results)} cases")
    print()
    print("| Metric | Value |")
    print("|---|---|")
    print(f"| Hit rate @ k | {report.hit_rate:.0%} |")
    print(f"| Strict pass rate (within documented `within_top`) | {report.strict_pass_rate:.0%} |")
    print(f"| MRR | {report.mrr:.3f} |")
    print()

    if report.failures:
        print("Cases not meeting their documented `within_top`:")
        print()
        for r in report.failures:
            print(f"- `{r.case.query}` (expected `{r.case.expected_doc_id}` "
                  f"within top {r.case.within_top})")
    else:
        print("All cases met their documented `within_top`.")


if __name__ == "__main__":
    main()

"""Run the co-pilot evaluation suite.

python -m evals                 # offline, deterministic (no Ollama needed)
python -m evals --live          # real Ollama models and Chroma index
python -m evals --case missing-evidence --json report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from evals.harness import LexicalRetriever, load_cases, run_case, summarise


async def _main(args: argparse.Namespace) -> int:
    cases = load_cases()
    if args.case:
        cases = [case for case in cases if case["id"] in set(args.case)]
    live_llm: Any = None
    min_score = 0.05
    if args.live:
        import httpx

        from app.config import get_settings
        from app.knowledge.embeddings import OllamaEmbeddings
        from app.llm import OllamaChatClient
        from app.retrieval import ChromaRetriever

        settings = get_settings()
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.llm_read_timeout_seconds, connect=5)
        )
        embeddings = OllamaEmbeddings(
            settings.embedding_model, settings.ollama_url, http_client=client
        )
        retriever: Any = ChromaRetriever(
            settings.resolve(settings.chroma_dir), embeddings.aembed_query, embeddings.model
        )
        live_llm = OllamaChatClient(
            settings.ollama_url,
            settings.llm_model,
            settings.llm_temperature,
            settings.llm_num_ctx,
            settings.llm_num_predict,
            http_client=client,
        )
        min_score = settings.min_retrieval_score
    else:
        retriever = LexicalRetriever()

    results = [
        await run_case(case, retriever=retriever, live_llm=live_llm, min_score=min_score)
        for case in cases
    ]
    for result in results:
        mark = "PASS" if result.passed else "FAIL"
        print(f"[{mark}] {result.category:<24} {result.case_id}")
        for check in result.checks:
            if not check.passed:
                print(f"        x {check.name} {check.detail}")
    summary = summarise(results)
    print(
        f"\n{summary['passed']}/{summary['cases']} cases passed "
        f"({'live' if args.live else 'offline'} mode)."
    )
    for name, bucket in sorted(summary["categories"].items()):
        print(f"  {name:<24} {bucket['passed']}/{bucket['cases']}")
    print(
        f"  answers with invalid citations: {summary['answers_with_invalid_citations']}"
        f"/{summary['answers_checked']}, unverified figures: "
        f"{summary['answers_with_unverified_figures']}/{summary['answers_checked']}, "
        f"weak citations: {summary['answers_with_weak_citations']}/{summary['answers_checked']}"
    )
    print("Automated checks are heuristics and do not guarantee factual accuracy.")
    if args.json:
        payload = {
            "summary": summary,
            "results": [
                {
                    "id": r.case_id,
                    "category": r.category,
                    "passed": r.passed,
                    "answer": r.answer,
                    "checks": [vars(check) for check in r.checks],
                }
                for r in results
            ],
        }
        args.report = payload
    return 0 if summary["passed"] == summary["cases"] else 1


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m evals")
    parser.add_argument("--live", action="store_true", help="Use Ollama and the Chroma index.")
    parser.add_argument("--case", action="append", help="Run only this case ID (repeatable).")
    parser.add_argument("--json", help="Write a JSON report to this path.")
    args = parser.parse_args()
    args.report = None
    status = asyncio.run(_main(args))
    if args.json and args.report is not None:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(args.report, handle, indent=2, ensure_ascii=False)
    sys.exit(status)


if __name__ == "__main__":
    main()

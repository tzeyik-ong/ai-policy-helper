#!/usr/bin/env python3
"""
Regression eval script — runs acceptance-check queries and asserts correct citations.

Usage:
    python scripts/eval.py                             # backend at http://localhost:8000
    python scripts/eval.py --base-url http://localhost:8000
    python scripts/eval.py --ingest                    # force re-ingest before running

Requires:
    pip install httpx   (or run inside the backend container)
"""

import argparse
import json
import sys

import httpx

CASES = [
    {
        "name": "Damaged blender return (acceptance check 1)",
        "query": "Can a customer return a damaged blender after 20 days?",
        "expected": ["Returns_and_Refunds", "Warranty_Policy"],
        "match": "all",  # all expected titles must appear in citations
    },
    {
        "name": "Shipping SLA East Malaysia (acceptance check 2)",
        "query": "What's the shipping SLA to East Malaysia for bulky items?",
        "expected": ["Delivery_and_Shipping"],
        "match": "any",  # at least one expected title must appear
    },
    {
        "name": "Product warranty coverage",
        "query": "What does the standard warranty cover for electronics?",
        "expected": ["Warranty_Policy"],
        "match": "any",
    },
    {
        "name": "Return window for small appliances",
        "query": "What is the return window for small appliances?",
        "expected": ["Returns_and_Refunds"],
        "match": "any",
    },
]


def _titles_match(expected: list, titles: set, match: str) -> bool:
    if match == "all":
        return all(any(e in t for t in titles) for e in expected)
    return any(any(e in t for t in titles) for e in expected)


def run_eval(base_url: str, do_ingest: bool = True) -> int:
    client = httpx.Client(base_url=base_url, timeout=120)
    failures = 0

    print(f"[eval] Backend: {base_url}")

    # Health check
    try:
        health = client.get("/api/health").json()
        print(f"[eval] Health: {health}")
    except Exception as e:
        print(f"[eval] ERROR: Cannot reach backend — {e}")
        return 1

    # Ingest
    if do_ingest:
        print("[eval] Ingesting docs …", flush=True)
        resp = client.post("/api/ingest")
        resp.raise_for_status()
        data = resp.json()
        print(f"[eval] Ingested: {data['indexed_docs']} docs, {data['indexed_chunks']} chunks")

    # Run queries
    print()
    for i, case in enumerate(CASES, 1):
        print(f"[{i}/{len(CASES)}] {case['name']}")
        print(f"  Query: {case['query'][:80]}")

        try:
            resp = client.post("/api/ask", json={"query": case["query"]})
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  ✗ FAIL — request error: {e}")
            failures += 1
            continue

        titles = {c["title"].replace(".md", "") for c in data.get("citations", [])}
        answer_preview = data.get("answer", "")[:100].replace("\n", " ")
        cached = data.get("cached", False)

        print(f"  Citations: {titles}")
        print(f"  Answer: {answer_preview}{'…' if len(data.get('answer',''))>100 else ''}")
        if cached:
            print("  (cached response)")

        if _titles_match(case["expected"], titles, case["match"]):
            print(f"  ✓ PASS")
        else:
            print(f"  ✗ FAIL — expected {case['match']}({case['expected']}), got {titles}")
            failures += 1
        print()

    # Metrics summary
    try:
        m = client.get("/api/metrics").json()
        print("[eval] Final metrics:")
        print(f"  Docs indexed:        {m['total_docs']}")
        print(f"  Chunks indexed:      {m['total_chunks']}")
        print(f"  Total queries run:   {m['query_count']}")
        print(f"  Avg retrieval:       {m['avg_retrieval_latency_ms']} ms")
        print(f"  Avg generation:      {m['avg_generation_latency_ms']} ms")
    except Exception as e:
        print(f"[eval] Could not fetch metrics: {e}")

    print()
    if failures == 0:
        print("✓ All eval cases passed.")
    else:
        print(f"✗ {failures}/{len(CASES)} eval case(s) FAILED.")

    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Citation accuracy regression eval")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Backend URL")
    parser.add_argument(
        "--no-ingest",
        action="store_true",
        help="Skip ingest step (use if docs are already indexed)",
    )
    args = parser.parse_args()
    sys.exit(run_eval(args.base_url, do_ingest=not args.no_ingest))

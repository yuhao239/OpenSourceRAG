"""
eval.py

A standalone, enhanced metrics runner for evaluating RAG pipelines.
This script focuses on providing robust, LLM-judged metrics for
faithfulness and correctness, targeting specific experimental comparisons.

Key Features:
- A/B/C Pipeline Testing: Compares No-RAG, RAG, and RAG+Verify.
- LLM-as-a-Judge: Uses a separate LLM to score faithfulness and correctness
  on a 1-5 scale for more nuanced evaluation than binary scores.
- Focused Dataset: Designed to be run with a smaller, high-quality dataset
  where answers are verifiably present in the source documents.
- Clear Summaries: Outputs clear, comparative summaries for the two key experiments:
  1. The value of RAG (Pipeline B vs. A)
  2. The impact of Verification (Pipeline C vs. B)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import dataclasses
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pathlib import Path
SRC_ROOT = Path(__file__).resolve().parents[2] / 'src'
# Ensure project modules under src/ are importable before any project imports
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Lazy imports of project modules to keep this script standalone
from config import Config
from events import StartQueryEvent
from work_flows.query_workflow import QueryWorkflow
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.llms.ollama import Ollama


# -----------------------------
# Data Structures
# -----------------------------

@dataclasses.dataclass
class QAItem:
    """Represents a single question-answer item for evaluation."""
    qid: str
    question: str
    gold_answer: Optional[str] = None
    gold_citations: Optional[List[str]] = None

    @staticmethod
    def from_jsonl_line(obj: Dict[str, Any]) -> "QAItem":
        return QAItem(
            qid=str(obj.get("qid", "unknown_qid")),
            question=str(obj["question"]).strip(),
            gold_answer=(obj.get("gold_answer") or None),
            gold_citations=list(obj.get("gold_citations") or []) or None,
        )


def load_dataset(path: Path) -> List[QAItem]:
    """Loads a dataset from a .jsonl file."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    items: List[QAItem] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                items.append(QAItem.from_jsonl_line(obj))
    return items


# -----------------------------
# Metric Calculation Utilities
# -----------------------------

def _dcg(scores: List[float]) -> float:
    """Discounted Cumulative Gain."""
    return sum((2**s - 1) / math.log2(i + 2) for i, s in enumerate(scores))


def ndcg_at_k(predicted_ids: List[str], gold_ids: Optional[List[str]], k: int = 10) -> Optional[float]:
    """Normalized Discounted Cumulative Gain @ k."""
    if not gold_ids:
        return None
    gold_set = set(str(g).strip() for g in gold_ids if str(g).strip())
    if not gold_set:
        return None

    relevance_scores = [1.0 if pid in gold_set else 0.0 for pid in predicted_ids[:k]]
    dcg = _dcg(relevance_scores)
    ideal_scores = [1.0] * min(len(gold_set), k)
    idcg = _dcg(ideal_scores)

    return (dcg / idcg) if idcg > 0 else 0.0


def summarize_latencies(values: List[float]) -> Dict[str, Optional[float]]:
    """Calculates p50, p95, and mean for a list of latency values."""
    if not values:
        return {"p50_ms": None, "p95_ms": None, "mean_ms": None}

    p50 = statistics.median(values)
    p95 = sorted(values)[int(len(values) * 0.95)] if len(values) > 1 else values[0]
    mean = statistics.mean(values)

    return {"p50_ms": round(p50), "p95_ms": round(p95), "mean_ms": round(mean)}


# -----------------------------
# LLM-as-a-Judge for Metrics
# -----------------------------

class LLMJudge:
    """Uses an LLM to provide scores for qualitative metrics."""

    def __init__(self, config: Config):
        self.llm = Ollama(
            model=config.UTILITY_MODEL,
            request_timeout=60.0,
            base_url=f"http://{config.OLLAMA_HOST}:11434",
        )

    async def judge_metric(self, prompt: str) -> Optional[int]:
        """Helper to run the LLM judge and parse its score.
        Prefer parsing an explicit "Score: X" pattern; fall back to first integer.
        """
        try:
            response = await self.llm.acomplete(prompt)
            text = (response.text or '').strip()
            import re
            m = re.search(r"Score\s*:\s*([1-5])\b", text)
            if m:
                return int(m.group(1))
            # Fallback: first integer between 1 and 5
            for word in text.split():
                if word.isdigit():
                    val = int(word)
                    if 1 <= val <= 5:
                        return val
            return None
        except Exception as e:
            print(f"LLM Judge failed: {e}")
            return None

    async def score_faithfulness(self, question: str, answer: str, context: str) -> int:
        prompt = f"""
        You are an impartial judge evaluating the faithfulness of an answer to its provided context.
        - The user asked: "{question}"
        - The system was given the following context: <context>{context}</context>
        - The system generated this answer: "{answer}"

        **Evaluation Task:**
        Compare the generated answer against the provided context. The answer must be 100% supported by the information in the context.

        **Scoring Rubric (1-5):**
        1: Hallucination. The answer contains significant information not present or contradicted by the context.
        3: Mostly Faithful. The answer is plausible but may infer minor details not explicitly stated.
        5: Perfectly Faithful. Every single claim in the answer is directly and explicitly supported by the provided context. It is acceptable for the answer to synthesize or rephrase information, as long as the core information of each claim can be found in the sources.

        Your response must end with "Score: [1-5]".
        """
        return await self.judge_metric(prompt)

    async def score_correctness(self, question: str, answer: str, gold_answer: str) -> int:
        prompt = f"""
        You are an impartial judge evaluating the correctness of an answer.
        - The user asked: "{question}"
        - The reference "gold" answer is: "{gold_answer}"
        - The system generated this answer: "{answer}"

        **Evaluation Task:**
        Compare the system's answer to the gold answer.

        **Scoring Rubric (1-5):**
        1: Incorrect. The answer is completely wrong.
        3: Partially Correct. The answer is on the right track but is missing key information or contains minor inaccuracies.
        5: Perfectly Correct. The answer is complete and fully aligns with the gold answer.

        Your response must end with "Score: [1-5]".
        """
        return await self.judge_metric(prompt)


# -----------------------------
# Pipeline Runners
# -----------------------------

class PipelineRunners:
    """Encapsulates the logic for running each evaluation pipeline (A, B, C)."""

    def __init__(self, config: Config):
        self.config = config

    def _get_files_from_sources(self, sources: List[Dict[str, Any]]) -> List[str]:
        """Extracts unique filenames from the sources list."""
        files = set()
        for s in sources or []:
            f = s.get("file") or s.get("path")
            if f:
                files.add(os.path.basename(str(f)))
        return sorted(list(files))

    async def run_A_no_rag(self, item: QAItem) -> Dict[str, Any]:
        """Pipeline A: No-RAG, direct LLM answer."""
        llm = Ollama(
            model=self.config.LLM_MODEL,
            request_timeout=self.config.LLM_REQUEST_TIMEOUT,
            base_url=f"http://{self.config.OLLAMA_HOST}:11434",
        )
        prompt = f"Question: {item.question}\nAnswer:"

        t0 = time.perf_counter()
        resp = await llm.acomplete(prompt)
        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "answer": resp.text.strip(),
            "retrieved_context": "",
            "retrieved_files": [],
            "used_retrieval": False,
            "retrieval_confidence": "none",
            "retrieval_confidence_features": {},
            "timings": {"total_workflow_ms": latency_ms},
        }

    async def run_B_rag_no_verify(self, item: QAItem) -> Dict[str, Any]:
        """Pipeline B: RAG without the verification step."""
        cfg = Config()
        cfg.USE_VERIFIER = False
        wf = QueryWorkflow(config=cfg, timeout=600)
        result = await wf.run(StartQueryEvent(query=item.question))

        sources = result.get("sources", [])
        context_str = "\n\n".join(
            [f"Source {i+1}:\n{s.get('highlight_text', '')}" for i, s in enumerate(sources)]
        )

        return {
            "answer": result.get("final_answer", ""),
            "retrieved_context": context_str,
            "retrieved_files": self._get_files_from_sources(sources),
            "used_retrieval": bool(sources),
            "retrieval_confidence": result.get("retrieval_confidence", None),
            "retrieval_confidence_features": result.get("retrieval_confidence_features", {}),
            "timings": {"total_workflow_ms": result.get("timings", {}).get("total_workflow", 0) * 1000},
        }

    async def run_C_rag_with_verify(self, item: QAItem) -> Dict[str, Any]:
        """Pipeline C: Full RAG pipeline with verification."""
        cfg = Config()
        cfg.USE_VERIFIER = True
        wf = QueryWorkflow(config=cfg, timeout=600)
        result = await wf.run(StartQueryEvent(query=item.question))

        sources = result.get("sources", [])
        context_str = "\n\n".join(
            [f"Source {i+1}:\n{s.get('highlight_text', '')}" for i, s in enumerate(sources)]
        )

        return {
            "answer": result.get("final_answer", ""),
            "retrieved_context": context_str,
            "retrieved_files": self._get_files_from_sources(sources),
            "used_retrieval": bool(sources),
            "retrieval_confidence": result.get("retrieval_confidence", None),
            "retrieval_confidence_features": result.get("retrieval_confidence_features", {}),
            "timings": {"total_workflow_ms": result.get("timings", {}).get("total_workflow", 0) * 1000},
        }


# -----------------------------
# Main Evaluation Orchestrator
# -----------------------------

async def run_evaluation_for_item(
    item: QAItem,
    pipeline_id: str,
    runners: PipelineRunners,
    judge: LLMJudge,
    run_id: str,
) -> Dict[str, Any]:
    """Runs a single QA item through a specified pipeline and calculates all metrics."""
    print(f"Running qid={item.qid}, pipeline={pipeline_id}...")

    if pipeline_id == "A":
        output = await runners.run_A_no_rag(item)
    elif pipeline_id == "B":
        output = await runners.run_B_rag_no_verify(item)
    elif pipeline_id == "C":
        output = await runners.run_C_rag_with_verify(item)
    else:
        raise ValueError(f"Unknown pipeline: {pipeline_id}")

    # Calculate metrics
    faithfulness: Optional[int] = None
    # Only score faithfulness when retrieval was used and context is non-empty
    if output.get("used_retrieval") and (output.get("retrieved_context") or "" ).strip():
        faithfulness = await judge.score_faithfulness(
            question=item.question,
            answer=output["answer"],
            context=output["retrieved_context"]
        )
    correctness = None
    if item.gold_answer:
        correctness = await judge.score_correctness(
            question=item.question,
            answer=output["answer"],
            gold_answer=item.gold_answer
        )
    ndcg = ndcg_at_k(output["retrieved_files"], item.gold_citations, k=10)

    # Assemble final record for logging
    record = {
        "run_id": run_id,
        "qid": item.qid,
        "pipeline": pipeline_id,
        "question": item.question,
        "answer": output["answer"],
        "latency_ms": output["timings"]["total_workflow_ms"],
        "retrieved_files": output["retrieved_files"],
        "used_retrieval": output.get("used_retrieval", False),
        "retrieval_confidence": output.get("retrieval_confidence"),
        "retrieval_confidence_features": output.get("retrieval_confidence_features", {}),
        "gold_citations": item.gold_citations,
        "faithfulness_score": faithfulness,
        "correctness_score": correctness,
        "ndcg_at_10": ndcg,
    }
    return record


def print_summary(results: List[Dict[str, Any]]):
    """Prints a formatted summary of the evaluation results."""
    by_pipeline: Dict[str, List[Dict[str, Any]]] = {"A": [], "B": [], "C": []}
    for r in results:
        if r.get("pipeline") in by_pipeline:
            by_pipeline[r["pipeline"]].append(r)

    def avg_metric(pipeline_id: str, metric: str) -> Optional[float]:
        values = [r[metric] for r in by_pipeline[pipeline_id] if r.get(metric) is not None]
        return round(statistics.mean(values), 2) if values else None

    print("\n\n" + "="*50)
    print(" " * 15 + "EVALUATION SUMMARY")
    print("="*50)

    # --- Experiment 1: Value of RAG (B vs. A) ---
    print("\n--- Experiment 1: The Value of RAG (Pipeline B vs. A) ---\n")
    if by_pipeline['A'] and by_pipeline['B']:
        a_correctness = avg_metric("A", "correctness_score")
        b_correctness = avg_metric("B", "correctness_score")
        print(f"  Correctness (1-5):")
        print(f"    - No-RAG (A):      {a_correctness or 'N/A'}")
        print(f"    - RAG (B):         {b_correctness or 'N/A'}")
        if a_correctness and b_correctness and b_correctness > a_correctness:
            print("  => Conclusion: RAG significantly improved answer correctness.")
        else:
            print("  => Conclusion: RAG did not show a clear improvement in correctness.")

        b_ndcg = avg_metric("B", "ndcg_at_10")
        print(f"\n  Retrieval Quality (nDCG@10):")
        print(f"    - RAG (B):         {b_ndcg or 'N/A'}")

    else:
        print("  Could not run comparison. Missing data for Pipeline A or B.")

    # --- Experiment 2: Impact of Verification (C vs. B) ---
    print("\n--- Experiment 2: The Impact of Verification (Pipeline C vs. B) ---\n")
    if by_pipeline['B'] and by_pipeline['C']:
        # Retrieval-only faithfulness
        b_faith_values = [r['faithfulness_score'] for r in by_pipeline['B'] if r.get('used_retrieval') and r.get('faithfulness_score') is not None]
        c_faith_values = [r['faithfulness_score'] for r in by_pipeline['C'] if r.get('used_retrieval') and r.get('faithfulness_score') is not None]
        b_faithfulness = round(statistics.mean(b_faith_values), 2) if b_faith_values else None
        c_faithfulness = round(statistics.mean(c_faith_values), 2) if c_faith_values else None
        print(f"  Faithfulness (1-5):")
        print(f"    - RAG (B):         {b_faithfulness or 'N/A'}")
        print(f"    - RAG+Verify (C):  {c_faithfulness or 'N/A'}")
        if c_faithfulness and b_faithfulness and c_faithfulness > b_faithfulness:
             print("  => Conclusion: The Verifier successfully improved the faithfulness of answers.")
        else:
             print("  => Conclusion: The Verifier did not show a clear improvement in faithfulness.")

        b_latency = summarize_latencies([r['latency_ms'] for r in by_pipeline['B'] if r.get('latency_ms') is not None])
        c_latency = summarize_latencies([r['latency_ms'] for r in by_pipeline['C'] if r.get('latency_ms') is not None])
        print(f"\n  Latency (p50):")
        print(f"    - RAG (B):         {b_latency['p50_ms']} ms")
        print(f"    - RAG+Verify (C):  {c_latency['p50_ms']} ms")
        if c_latency['p50_ms'] and b_latency['p50_ms']:
            latency_cost = c_latency['p50_ms'] - b_latency['p50_ms']
            print(f"  => Conclusion: Verification added an average of {latency_cost} ms to the response time.")

    else:
        print("  Could not run comparison. Missing data for Pipeline B or C.")

    print("\n" + "="*50 + "\n")

    # --- Retrieval Confidence Buckets (diagnostics) ---
    def _bucket_report(items: List[Dict[str, Any]], label: str):
        if not items:
            print(f"(no items for {label})")
            return
        buckets = {}
        for r in items:
            b = r.get('retrieval_confidence') or 'unknown'
            buckets.setdefault(b, []).append(r)
        print(f"Retrieval Confidence Buckets — {label}")
        for bkey in ['none','low','mid','high','unknown']:
            arr = buckets.get(bkey) or []
            if not arr:
                continue
            # Faithfulness averaged only where used_retrieval and a score exists
            fvals = [x.get('faithfulness_score') for x in arr if x.get('used_retrieval') and x.get('faithfulness_score') is not None]
            favg = round(statistics.mean(fvals), 2) if fvals else None
            print(f"  - {bkey}: count={len(arr)}; faithfulness(avg, retr-only)={favg if favg is not None else 'N/A'}")

    print("\n--- Retrieval Confidence Buckets (B) ---")
    _bucket_report(by_pipeline['B'], 'B')
    print("\n--- Retrieval Confidence Buckets (C) ---")
    _bucket_report(by_pipeline['C'], 'C')


async def main():
    """Main function to parse arguments and run the evaluation."""
    parser = argparse.ArgumentParser(description="Enhanced RAG Metrics Runner")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the high-quality .jsonl dataset.")
    parser.add_argument("--pipelines", type=str, default="A,B,C", help="Comma-separated list of pipelines to run (A,B,C).")
    parser.add_argument("--run-id", type=str, default=f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}", help="A unique ID for this evaluation run.")
    parser.add_argument("--concurrency", type=int, default=2, help="Number of concurrent evaluation tasks.")
    args = parser.parse_args()

    # Setup
    config = Config()
    judge = LLMJudge(config)
    runners = PipelineRunners(config)
    dataset = load_dataset(Path(args.dataset))
    pipelines_to_run = [p.strip().upper() for p in args.pipelines.split(",")]

    # Create output file
    # Prefer experiments/logs, fallback to metrics_logs for backward compatibility
    preferred = Path("./experiments/logs")
    legacy = Path("./metrics_logs")
    out_dir = preferred if preferred.exists() or not legacy.exists() else legacy
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.run_id}.jsonl"

    # Run evaluations concurrently
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = []

    async def task_wrapper(item, pipeline_id):
        async with semaphore:
            return await run_evaluation_for_item(item, pipeline_id, runners, judge, args.run_id)

    for item in dataset:
        for pipeline_id in pipelines_to_run:
            tasks.append(task_wrapper(item, pipeline_id))

    results = await asyncio.gather(*tasks)

    # Save results and print summary
    with out_path.open("w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

    print(f"\nEvaluation complete. Full results saved to: {out_path}")
    print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())

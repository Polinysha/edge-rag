import argparse
import asyncio
import csv
import os

from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRelevance,
    Faithfulness,
)

from src.llm.llm_client import async_client as groq_async_client, MODEL_NAME
from src.pipeline.retrieval import search
from src.pipeline.generation import generate
from src.pipeline.graph import build_graph
from data.eval_dataset import EVAL_QUESTIONS

MAX_RETRIES_PER_METRIC = 3
BASELINE_TOP_K = 5
METRIC_NAMES = ("context_relevance", "faithfulness", "answer_relevance", "context_precision")

RESULTS_CSV = "data/eval_results.csv"
COMPARISON_MD = "data/eval_comparison.md"

# Инициализируем LLM для RAGAS через новый llm_factory, передавая ему наш асинхронный клиент Groq
_ragas_llm = llm_factory(
    model=MODEL_NAME,
    provider="openai",
    client=groq_async_client,
    max_tokens=2000
)

_ragas_embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

_faithfulness = Faithfulness(llm=_ragas_llm)
_context_relevance = ContextRelevance(llm=_ragas_llm)
_answer_relevancy = AnswerRelevancy(llm=_ragas_llm, embeddings=_ragas_embeddings)
_context_precision = ContextPrecisionWithReference(llm=_ragas_llm)


async def _score_with_retries(coro_factory, metric_name: str) -> float:
    last_error = None
    for attempt in range(MAX_RETRIES_PER_METRIC):
        try:
            result = await coro_factory()
            return result.value
        except Exception as e:  # noqa: BLE001 - flaky free-tier metric calls
            last_error = e
            # Ожидаем 8 секунд перед следующей попыткой, чтобы лимиты токенов сбросились
            print(f"    ! {metric_name} пауза {attempt + 1}/{MAX_RETRIES_PER_METRIC} (ждём 8 сек из-за лимитов...)")
            await asyncio.sleep(8)

    print(f"    ! {metric_name} failed after {MAX_RETRIES_PER_METRIC} attempts "
          f"({last_error}); falling back to 0.0")
    return 0.0


async def _score_triple(question: str, answer: str, contexts: list[str], reference: str) -> dict:
    # Запускаем метрики СТРОГО по очереди с паузами, чтобы не пробить лимит в 6000 токенов в минуту

    faithfulness = await _score_with_retries(
        lambda: _faithfulness.ascore(
            user_input=question, response=answer, retrieved_contexts=contexts
        ),
        "faithfulness",
    )
    await asyncio.sleep(4)  # Пауза для сброса счетчика TPM (Tokens Per Minute)

    context_relevance = await _score_with_retries(
        lambda: _context_relevance.ascore(
            user_input=question, retrieved_contexts=contexts
        ),
        "context_relevance",
    )
    await asyncio.sleep(4)

    answer_relevance = await _score_with_retries(
        lambda: _answer_relevancy.ascore(user_input=question, response=answer),
        "answer_relevance",
    )
    await asyncio.sleep(4)

    context_precision = await _score_with_retries(
        lambda: _context_precision.ascore(
            user_input=question, reference=reference, retrieved_contexts=contexts
        ),
        "context_precision",
    )

    return {
        "context_relevance": context_relevance,
        "faithfulness": faithfulness,
        "answer_relevance": answer_relevance,
        "context_precision": context_precision,
    }


def run_baseline_arm(question: str) -> tuple[str, list[str]]:
    chunks = search(question, top_k=BASELINE_TOP_K)
    answer = generate(question, chunks)
    contexts = [c["text"] for c in chunks]
    return answer, contexts


def run_graph_arm(graph, question: str) -> tuple[str, list[str]]:
    state = graph.invoke({"question": question, "retry_count": 0})
    answer = state.get("generation", "")
    contexts = [doc["text"] for doc in state.get("documents", [])]
    return answer, contexts


def aggregate(rows: list[dict], prefix: str) -> dict:
    means = {}
    for metric in METRIC_NAMES:
        values = [r[f"{prefix}_{metric}"] for r in rows]
        means[metric] = sum(values) / len(values) if values else 0.0
    return means


def build_comparison_md(baseline_means: dict, graph_means: dict, n: int) -> str:
    lines = [
        "# Task 6 — RAGAS: baseline (Task 3) vs graph (Task 5)",
        "",
        f"Evaluated on {n} question/ground_truth pairs from `data/eval_dataset.py`.",
        "",
        "| Metric | Было (Task 3) | Стало (Task 5) | Δ |",
        "|---|---|---|---|",
    ]
    for metric in METRIC_NAMES:
        before = baseline_means[metric]
        after = graph_means[metric]
        delta = after - before
        arrow = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "≈")
        lines.append(
            f"| {metric} | {before:.3f} | {after:.3f} | {arrow} {delta:+.3f} |"
        )
    lines += [
        "",
        "> Note: a 0.0 on any single question can mean the RAGAS metric call "
        "failed (free-tier flakiness), not that the answer was bad — see "
        "`data/eval_results.csv` for per-question values.",
    ]
    return "\n".join(lines)


def write_results_csv(rows: list[dict], path: str = RESULTS_CSV) -> None:
    fieldnames = ["question", "ground_truth", "baseline_answer", "graph_answer"]
    for arm in ("baseline", "graph"):
        for metric in METRIC_NAMES:
            fieldnames.append(f"{arm}_{metric}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Task 6 RAGAS evaluation")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N questions (smoke run)")
    args = parser.parse_args()

    dataset = EVAL_QUESTIONS[: args.limit] if args.limit else EVAL_QUESTIONS
    graph = build_graph()

    rows = []
    failed_questions = []
    for i, item in enumerate(dataset, start=1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        print(f"[{i}/{len(dataset)}] {question}")

        try:
            baseline_answer, baseline_ctx = run_baseline_arm(question)
        except Exception as e:  # noqa: BLE001
            print(f"    ! baseline arm failed: {e}")
            baseline_answer, baseline_ctx = "[PIPELINE ERROR]", []

        try:
            graph_answer, graph_ctx = run_graph_arm(graph, question)
        except Exception as e:  # noqa: BLE001
            print(f"    ! graph arm failed: {e}")
            graph_answer, graph_ctx = "[PIPELINE ERROR]", []

        if baseline_answer == "[PIPELINE ERROR]" or graph_answer == "[PIPELINE ERROR]":
            failed_questions.append(question)

        baseline_scores = asyncio.run(
            _score_triple(question, baseline_answer, baseline_ctx, ground_truth)
        )

        # Добавляем небольшую паузу между оценкой бейзлайна и графа
        print("    --- Переход к оценке графа, небольшая пауза ---")
        asyncio.run(asyncio.sleep(4))

        graph_scores = asyncio.run(
            _score_triple(question, graph_answer, graph_ctx, ground_truth)
        )

        row = {
            "question": question,
            "ground_truth": ground_truth,
            "baseline_answer": baseline_answer,
            "graph_answer": graph_answer,
        }
        for metric in METRIC_NAMES:
            row[f"baseline_{metric}"] = baseline_scores[metric]
            row[f"graph_{metric}"] = graph_scores[metric]
        rows.append(row)

        write_results_csv(rows)

    baseline_means = aggregate(rows, "baseline")
    graph_means = aggregate(rows, "graph")

    write_results_csv(rows)
    comparison = build_comparison_md(baseline_means, graph_means, len(rows))
    with open(COMPARISON_MD, "w", encoding="utf-8") as f:
        f.write(comparison)

    print("\n" + comparison)
    if failed_questions:
        print(f"\n! {len(failed_questions)} question(s) had a pipeline failure "
              f"(marked [PIPELINE ERROR] in the CSV):")
        for q in failed_questions:
            print(f"  - {q}")
    print(f"\nSaved per-question results to {RESULTS_CSV}")
    print(f"Saved comparison table to {COMPARISON_MD}")


if __name__ == "__main__":
    main()
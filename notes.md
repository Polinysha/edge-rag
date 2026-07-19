# Baseline observations (Task 3)

Notes on weak or broken answers from `data/baseline.csv`, with a guess at the cause for each.
Nothing here gets fixed yet — this is just the diagnosis pass before deciding what to improve.

## "Who are the parties to the agreement?"

Answer: `User Safety: unsafe Safety Categories: PII/Privacy`

Cause: generation, not retrieval. This isn't a real answer — it looks like the free-tier
router picked a model with a safety/moderation layer that flagged the question as touching
personal data (the contract's signatories) and returned a moderation label instead of
calling through to the actual completion. The retrieved chunks were probably fine; the LLM
call itself failed to produce real output. Worth re-running this exact question a few times
to see if it's the specific model OpenRouter's free router picked this time, or something
that happens consistently with this kind of question.

## "What must be prepared at the end of the internship?"

Answer: `User Safety: safe`

Cause: generation. Same symptom as above, different content — just confirms this is some kind
of moderation/safety-check response being returned in place of a real answer, not something
specific to PII. Likely the same root cause: an unstable free model behind the router.

## "Which institution is named in the document?"

Answer: says no institution name is in the excerpt, the field is left blank in the text.

Cause: this one might actually be correct, not a bug. Looking back at the raw extracted text
from `extract_pdf`, the contract template does have blank lines for institution names in some
places (it's a template with blanks to fill in, like "20__ г. №______"). Need to check the
actual chunk that got retrieved before assuming this is a retrieval miss — it's possible the
model is accurately describing a genuinely blank part of the template.

## "What kind of document is this, in legal terms?" / "What does the host organization need to monitor during the internship?" / "What kind of students does this agreement cover?"

Answers: all say the context doesn't contain enough information to answer.

Cause: likely retrieval, not generation. The phrasing of these questions doesn't closely
match the document's wording (e.g. "in legal terms" vs the document just saying "ДОГОВОР"),
so the relevant chunks may not be ranking high enough in the top-5 results. This is exactly
the kind of gap query expansion (Task 5) is meant to close — rephrasing the question to match
the document's actual terminology before searching again.

## General takeaway

Two distinct problem categories showed up here, and they need different fixes:

1. Generation reliability — the free model router occasionally returns safety/moderation
   stubs instead of real answers. This isn't something retrieval or chunking can fix.
   Possible mitigations later: retry on a malformed response, or pin to a specific model
   instead of the free router once a stable one is found.
2. Retrieval misses on questions phrased differently than the source text — exactly what
   query expansion and the grade_documents retry loop (both coming in Task 4-5) are meant
   to address.

# RAGAS evaluate node observations (Task 5)

## Free models and structured output don't mix reliably

Building the `evaluate` node (RAGAS scoring) surfaced the same free-tier instability from a
different angle. RAGAS's metrics use the `instructor` library to force the LLM into returning
JSON matching a Pydantic schema. Across a single debugging session:

- `openrouter/free` (the random router) returned truncated/invalid JSON for the faithfulness
  metric's statement-generation step.
- Pinning to `openai/gpt-oss-120b:free` (a reasoning model) "fixed" the JSON shape but leaked
  stray whitespace tokens from its chain-of-thought into the structured output, breaking
  parsing anyway — and then hit an upstream rate limit on retry.
- Pinning to `meta-llama/llama-4-scout:free` failed immediately: the model lost its free-tier
  status between when this was written and when it was tested. Same failure mode as
  `llama-3.1-8b:free` in Task 3 — free model slugs on OpenRouter are not stable over time.

Settled on: keep using the same `openrouter/free` router as the rest of the pipeline (for
consistency and because pinning specific slugs has failed twice now), but wrap each of the
three RAGAS metric calls in a retry loop (3 attempts) with a graceful fallback to a score of
0.0 if all attempts fail. This means one flaky metric doesn't crash the whole `evaluate` node
— the other two metrics still get scored and logged.

In one real run, `faithfulness` failed all 3 attempts (same truncated-JSON issue) and fell
back to 0.0, while `context_relevance` (1.0) and `answer_relevance` (0.58) succeeded normally.
The 0.0 faithfulness score here is a measurement failure, not a real signal that the answer
was unfaithful to the context — worth keeping in mind when looking at aggregate RAGAS numbers
later (Task 6): a 0.0 doesn't always mean "bad answer", sometimes it means "metric call failed".


---

## Task 6 — RAGAS Evaluation Dataset: ЗАКРЫТ ✅

**Выполнено:**

1. ✅ **Расширенный датасет** — 27 пар question/ground_truth в `data/eval_dataset.py`
   - 12 оригинальных вопросов из Task 3
   - 15 дополнительных вопросов про тот же документ

2. ✅ **RAGAS evaluation pipeline** — `src/pipeline/run_eval.py`
   - Сравнивает baseline (Task 3) vs graph (Task 5)
   - 4 метрики: context_relevance, faithfulness, answer_relevance, context_precision
   - Retry-логика для нестабильных free-tier LLM вызовов

3. ✅ **Hybrid search** — dense + sparse с RRF в `retrieve_node.py`
   - Dense: Qdrant query по вектору
   - Sparse: BM25 через rank_bm25
   - RRF merge с k=60

4. ✅ **Результаты оценки** (13 вопросов, без PIPELINE ERROR):
   - context_relevance: 0.556
   - faithfulness: 0.329
   - answer_relevance: 0.195

5. ✅ **Анализ слабых запросов** — `data/weak_queries_analysis.md`
   - Выделены худшие 20% по context_relevance
   - Диагностированы проблемы (retrieval не находит релевантные чанки)
   - Рекомендации для Task 7

6. ✅ **Thresholds для регрессии** — `tests/e2e/thresholds.json`
   ```json
   {
     "context_relevance": 0.506,
     "faithfulness": 0.279,
     "answer_relevance": 0.145
   }
   ```

7. ✅ **E2E regression test** — обновлён `tests/e2e/test_graph.py`
   - Новый тест `test_graph_metrics_above_thresholds()`
   - Проверяет метрики против порогов с допуском 0.05

**Известные ограничения:**

- Низкая answer_relevance (0.195) — модель часто говорит "answer is not in context"
- Много 0.0 метрик из-за нестабильности free-tier LLM
- Не все 27 вопросов были успешно прогнаны (некоторые упали с PIPELINE ERROR)

**Следующие шаги (Task 7):**

1. Parent-Child chunking — индексировать маленькие чанки, отдавать большие блоки
2. Структурный детект — резать по заголовкам PDF
3. Reranker — cross-encoder для точного ранжирования

Ожидаемый эффект: рост context_relevance до 0.7+ и reduction "answer is not in context" ответов.

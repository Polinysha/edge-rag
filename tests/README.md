# Testing strategy

This project splits tests into three layers, each answering a different question.

## `tests/unit/`

Tests individual functions and (later) individual graph nodes in isolation: extraction,
chunking, embedding, retrieval, generation. These check that the code does what it claims
to do — correct return shapes, correct boundary handling, correct metadata propagation.
Where a test would otherwise depend on a live external resource (Qdrant, OpenRouter), it
either mocks that dependency or, in the retrieval tests, spins up a disposable test collection
instead of touching production data.

## `tests/api/`

Tests the API **contract**: status codes, response shape, required fields, error handling
on malformed input. These tests do **not** check whether an answer is any good — that's not
what they're for. `/upload` and `/ask` are tested with the underlying pipeline functions
mocked out, so this suite runs in seconds and doesn't need a live Qdrant instance or an
OpenRouter API key. If `/ask` returns 200 with an `answer` field, the test passes — whether
that answer is actually correct is a separate concern, tested elsewhere.

The reasoning for keeping this strict: the pipeline behind `/ask` is going to change shape
several times (plain search+generate now, a LangGraph graph after Task 5, then hybrid search,
reranking, and so on after that). The request/response contract is what's supposed to stay
stable through all of that. Testing the contract separately from the pipeline internals means
those internal changes don't require touching the API tests at all.

## `tests/e2e/`

Not built yet. Once the LangGraph graph exists, this is where end-to-end tests against a
golden dataset will live, using RAGAS to score context relevance, faithfulness, and answer
relevance. This is the layer that actually judges answer quality — on purpose, it's separate
from both of the layers above.

## In short

If you're trying to find a bug, the rule of thumb is: a broken status code or malformed
response → look in `tests/api`. A function doing the wrong thing internally → `tests/unit`.
An answer that's technically well-formed but wrong or unhelpful → that's a `tests/e2e` /
RAGAS problem, not something the unit or API suites are meant to catch.

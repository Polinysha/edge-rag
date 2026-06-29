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

# Design decisions

Each entry gives the decision, the alternative considered, and why.

## 1. The access filter is part of the store contract, not a step in a pipeline

Every search method requires a `Principal`, and a store never returns a chunk
that principal cannot see. The alternative is to retrieve first and filter in
the pipeline. That makes access control something each new pipeline must
remember to do, and one that forgets leaks. With the filter in the contract, a
pipeline cannot be written that searches without it.

## 2. Filtering inside the query, with post-filtering kept only as a baseline

Filtering after ranking returns fewer than k results whenever restricted chunks
rank highly, so the people with the least access get the worst retrieval. The
naive pipeline keeps post-filtering so that cost can be measured. Even there the
visibility test runs in the store, so restricted content still never leaves it.

## 3. `access_denied` reveals that something exists, so it is a setting

Telling an asker "you do not have access" confirms that restricted material
matches their question. Some organisations want that (it tells people to request
access); others treat existence as confidential. `reveal_restricted_existence`
chooses. The check that distinguishes the two verdicts receives only a list of
booleans from the store: whether each top-ranked candidate is visible. No id,
text or score of a restricted chunk is exposed to make the decision.

## 4. The verdict is decided by sufficiency, then access, then verification

A question first gets whatever the asker may see. Only if that is insufficient
does the access check run. An answer that is drafted but not supported by its
citations becomes `verification_failed`, a different outcome from having no
evidence, and counted separately in evaluation.

## 5. Superseded documents are excluded from retrieval, not left to the model

Two versions of a procedure differ by one number. Asking a model to prefer "the
one marked current" works most of the time. Excluding superseded documents in
the query works every time. The baseline does not do this, which is one of the
things it is there to show.

## 6. Rank fusion without a model reranker by default

Reciprocal rank fusion is deterministic, free and needs no second provider. A
reranker is a slot (`Reranker` protocol) rather than a default, to be added when
an evaluation run shows it earns its cost.

## 7. Exact identifiers are indexed separately in Postgres

Postgres full-text ranking has no term-rarity weighting and its parser splits
`DEV-2025-033` into fragments. On questions that name a record, built-in
full-text search scored 0.53 recall where BM25 scored 0.94. Indexing record
identifiers in their own column and boosting exact matches closed the gap.

## 8. Generated SQL is parsed, not pattern-matched

The guard parses the statement and checks the tree: one statement, a SELECT, no
write or DDL node anywhere, only allow-listed tables and functions, a literal
row limit. String checks for words such as DROP are easy to evade and reject
legitimate queries. The connection is also read-only, so a guard bug is not
enough on its own to change data.

## 9. Rejected SQL is an execution failure

When the guard blocks a query the answer is `execution_failed`, not a polite
decline. The model tried to do something it must not; hiding that as "no answer
available" would make the system look better behaved than it was.

## 10. The agent gathers evidence; it does not write the answer

The planning model chooses searches, reads and queries. What it collects goes
through the same generation and verification as every other pipeline. An agent
that both gathers and asserts has nothing checking it.

## 11. Graph walks obey access control at every hop

A hop passes through a chunk. If the asker cannot see that chunk, the hop does
not exist for them. Otherwise a restricted document could connect two public
records and the connection itself would be a leak.

## 12. Redaction happens when a trace is written

A trace event has a public payload and an optional admin payload, stored apart.
The public payload is built only from visible hits and counts. Redacting at
display time means one rendering bug exposes restricted content; redacting at
write time means the content was never in the public record.

## 13. Receipts hash the event, not just link to each other

Each receipt hash covers the previous hash and the full event. A chain that only
links receipt to receipt verifies even if an event's content was edited.

## 14. Evaluation thresholds are counts

With three to eight questions per type, "90 percent" is not a meaningful
threshold. The gate says "10 of 12" and "6 of 6".

## 15. A correct figure with none of the right evidence is scored wrong

A document pipeline can produce the right number for a counting question by
coincidence. If nothing it retrieved is among the question's supporting
evidence, the answer is graded wrong and flagged as coincidental.

## 16. The offline generator and verifier are stand-ins and say so

They exist so tests and CI need no network. Their limits are documented where
they are defined, the offline run is expected to fail the gate, and the README
lists what offline results can and cannot show.

# Overnight results

This report is updated as the Option C research program finishes each
controlled stage. Development numbers are mean MAP@12 on the 2020-08-31 and
2020-09-07 2K folds. The reserved 8,000-customer 2020-09-16..22 holdout is not
inspected until `BEST_SYSTEM` is frozen.

## Evaluation integrity

The dataset ends on 2020-09-22. There is no unused calendar week.

Previously inspected “final” metrics used 2020-09-16 through 2020-09-22 on
overlapping 100 / 500 / 1K / 2K / 5K customer draws. Those numbers are
contaminated reference measurements, not a pristine test.

Official final evaluation: **customer holdout**. 10,610 validation-week buyers
were already seen. 58,374 were unused. 8,000 unused IDs are reserved in
`artifacts/overnight/holdout_customer_ids.txt`.

See `docs/evaluation_ledger.md`.

## Frozen prior conclusions

These stand unless a new controlled development result overturns them:

- LightGBM LambdaRank is the strongest ranker family so far.
- DCN V2 overfit and is not being blindly retried.
- Metadata two-tower is complementary and budgeted at K=50.
- Generic image-only DINOv2 retrieval is rejected.
- Text has retrieval signal but did not consistently improve final MAP.
- History-based two-tower routing is rejected.

## Baseline

`BASELINE_RANKER` is frozen under the development protocol.

```text
Popularity + Repeat + PMI + ALS + metadata content + metadata two-tower K=50
        ↓
LightGBM LambdaRank (300 trees, lr=0.05, 63 leaves)
        ↓
Top 12
```

| Fold | MAP@12 | Recall@12 | Hit Rate@12 | Train positives |
|---|---:|---:|---:|---:|
| 2020-08-31 | 0.02638 | 0.04143 | 0.1135 | 3,675 |
| 2020-09-07 | 0.02776 | 0.04468 | 0.1170 | 5,476 |
| **Mean** | **0.02707** | | | |

Candidate-pool oracle MAP@12 is about 0.29 on these 2K development folds, so most of the remaining gap is ranking and irreducible next-week uncertainty, not missing retrieval of an already-found item. Historical 5K Sep 16–22 MAP@12 ≈ 0.028 is a contaminated reference only.

## Ranking failure analysis

On retrieved positives across the two development folds (n=3,576):

| Slice | n | Median rank | Top-12 rate |
|---|---:|---:|---:|
| All retrieved positives | 3,576 | 101 | 15.6% |
| Exact repeat | 538 | 6 | 61.3% |
| Same product type, new article | 2,326 | 129 | 7.6% |
| 1 retrieval source | 2,433 | 124 | 9.9% |
| 3 sources | 253 | 32 | 37.9% |
| Head popularity | 1,187 | 64 | 18.4% |
| Long tail | 1,196 | 218 | 12.9% |
| Repeat-only provenance | 225 | 6 | 61.8% |
| ALS-only | 447 | 196 | 1.8% |
| PMI-only | 564 | 199 | 1.1% |
| Two-tower-only | 44 | 264 | 2.3% |

LambdaRank already does the easy job: repeats and multi-source popular items. It systematically buries complementary-retriever positives. Buried long-tail positives and high-ranked long-tail negatives are barely separable on the current features (similar source count and ALS scores; type affinity is only a weak margin). That argues for more supervision and harder negatives, not a larger neural ranker.

## Experiment timeline

| Stage | Status |
|---|---|
| Evaluation ledger | done |
| BASELINE_RANKER | done, mean MAP@12 0.02707 |
| Ranking failure analysis | done |
| Scaled supervision | queued behind LightGBM tune |
| Hard negatives | queued |
| Targeted crosses | queued |
| LightGBM tune | running (truncation trials worse than baseline so far) |
| CatBoost | queued |
| Listwise reranker | queued |
| GenRec-style | only if justified |
| Customer-holdout final | after freeze |

LightGBM tune so far, development mean MAP@12:

| Trial | Mean MAP@12 |
|---|---:|
| baseline | 0.02707 |
| lambdarank_truncation_level=12 | 0.02313 |
| lambdarank_truncation_level=20 | 0.02538 |
| lambdarank_truncation_level=30 | 0.02707 |
| num_leaves=31 | 0.02730 |
| num_leaves=127 | 0.02494 |
| min_child_samples=20 | 0.02600 |
| min_child_samples=100 | 0.02639 |
| lr=0.03, 500 trees | 0.02799 |
| lr=0.10 / xendcg / L2 | running |

`lr=0.03` with 500 trees is the first material gain: **+0.00092** over baseline, well above the 0.0002 keep-simpler band. `num_leaves=31` (+0.00023) is inside the noise of the baseline fold std (0.00069). Remaining trials must finish before promotion. Truncation, 127 leaves, and both `min_child` settings have lost.

## Product architecture

```text
                         USER
                          │
          ┌───────────────┴────────────────┐
          │                                │
          ▼                                ▼
   PERSONALIZED HOME                    SEARCH
          │                                │
   User history                     Query intent parser
          │                                │
          ▼                                ▼
 Multi-source retrieval          BM25 + semantic + attrs
          │                                │
          ▼                                ▼
      LambdaRank                      merge / RRF
          │                                │
          ▼                                ▼
       Top 12                     query relevance rank
                                           │
                                           ▼
                                  small personalization
                                           │
                                           ▼
                                      search results
```

Home recommendations and search are separate systems. Search is evaluated
only with a synthetic/structured benchmark because H&M has no query logs.

Lexical + structured synthetic eval (`scripts/evaluate_search.py --lexical-only`):

- Exact `{color} {type}` queries score 1.0 Precision@10 / NDCG@10 / MRR. That is
  expected: structured retrieval returns the attribute intersection used as
  labels. It is not real search quality.
- Curated style queries (token overlap labels, 9 scored): Precision@10 0.667,
  Recall@50 0.711, NDCG@10 0.775, MRR 0.788. `linen summer shirt` surfaces
  linen shirts via BM25; `black oversized hoodie` returns black hoodies, with
  oversized names ranked first.
- Semantic MiniLM retrieval is implemented against
  `artifacts/text_retrieval/text_embeddings.f32.npy` and is optional at serve
  time (`SEARCH_SEMANTIC=0` skips the encoder). Full hybrid numbers wait until
  ranking jobs release RAM.
- A weakly supervised search LambdaRank exists (`scripts/run_search_ranker.py`)
  and will be kept only if it beats this hybrid on held-out synthetic queries.

## Rejected ideas (already decided, not rerun)

These were tested before this overnight run. Stage 1 did not produce a new
hypothesis that would reopen them.

| Idea | Evidence | Decision |
|---|---|---|
| DCN V2 / MLP ranker | LambdaRank ~0.0275 vs MLP 0.0227 vs DCN 0.0148 on the earlier protocol | Reject. Implementation checks passed; generalization failed. |
| Image-only DINOv2 retrieval | Weak marginal recall; final MAP fell | Reject generic averaged image retrieval. |
| Unrestricted text candidate expansion | TT Recall@500 rose 0.0733→0.1479, but MAP did not; new positives sat at median rank ~501 | Freeze text expansion. |
| History-based two-tower routing | Selected against on earlier folds | Reject. |
| Treating Sep 16–22 as an unseen final week | Dataset ends 2020-09-22; those metrics were already inspected | Customer-holdout protocol instead. |

## Best system

Not frozen yet. Current champion remains BASELINE_RANKER. `num_leaves=31` is a
provisional challenger at 0.02730 and is not promoted until the remaining
LightGBM trials and later stages finish.

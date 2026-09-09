# Overnight results

This report is updated as the Option C research program finishes each
controlled stage. Development numbers are mean MAP@12 on the 2020-08-31 and
2020-09-07 2K folds. The reserved 8,000-customer 2020-09-16..22 holdout is not
inspected until `BEST_SYSTEM` is frozen.

## Executive summary

Current development champion is **CatBoost YetiRank** on the same six-source
features and 500-iteration / `lr=0.03` schedule. Mean MAP@12 is **0.02987**
versus LightGBM `lr03_n500` **0.02799** (+0.00188) and versus the 300-tree
LightGBM baseline **0.02707** (+0.00280, +10.3% relative). Both selection
folds moved up. The LightGBM slower schedule was a real but smaller lift.
Truncation, XENDCG, hard negatives, crosses, and extra weeks lost or tied.
Listwise aborted with signal 11 after the sanity check (OOM on five loaded
snapshots) and is rejected. Inverse-sqrt popularity weights scored 0.02743 vs unweighted 0.02799 and
are rejected. Group-size weights raised the mean to 0.02832 but dropped
the Sep 7 fold from 0.02811 to 0.02691, so they are rejected as not
fold-consistent. `BEST_SYSTEM` is frozen as six-source CatBoost YetiRank with no extra
weighting. Official customer-holdout MAP@12 on 8,000 unused 2020-09-16..22
buyers is **0.03068** (Recall@12 0.0538, Hit Rate@12 0.1375). That is +13.3%
relative to the development 300-tree baseline (0.02707) and slightly above
the CatBoost development mean (0.02987). This is a reserved-customer test
on an already-seen calendar week, not an unseen temporal window.

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

- LightGBM LambdaRank was the strongest ranker family before this run.
  Overnight CatBoost YetiRank beat it on both development folds.
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
| LightGBM tune | **selected `lr03_n500`, mean MAP@12 0.02799** |
| Hard negatives | **reject**; all-candidates 0.02799 vs best downsample 0.02204 |
| Targeted crosses | **reject**; 0.02606 vs baseline features 0.02799 |
| Scaled supervision | **reject**; 0.02809 vs current 0.02799 (+0.00010 < 0.0002) |
| CatBoost YetiRank | **keep**; 0.02987 vs LightGBM 0.02799 |
| Listwise reranker | **reject**; signal 11 / OOM after sanity check |
| Long-tail weights | **reject**; 0.02743 vs unweighted 0.02799 |
| Group-size / active-user weights | **reject**; mean +0.00033 but Sep 7 fold fell |
| GenRec-style | not justified by failure analysis |
| BEST_SYSTEM freeze | **frozen_for_holdout**: CatBoost YetiRank, four weeks |
| Customer-holdout final | **0.03068 MAP@12** on 8,000 unused buyers |

Official LightGBM tune, development mean MAP@12:

| Trial | Mean MAP@12 | Decision |
|---|---:|---|
| baseline (300 / 0.05 / 63) | 0.02707 | control |
| lambdarank_truncation_level=12 | 0.02313 | reject |
| lambdarank_truncation_level=20 | 0.02538 | reject |
| lambdarank_truncation_level=30 | 0.02707 | reject (tie; keep default) |
| num_leaves=31 | 0.02730 | reject (+0.00023 inside fold noise) |
| num_leaves=127 | 0.02494 | reject |
| min_child_samples=20 | 0.02600 | reject |
| min_child_samples=100 | 0.02639 | reject |
| **lr=0.03, 500 trees** | **0.02799** | **keep** |
| lr=0.10, 200 trees | 0.02382 | reject |
| rank_xendcg | 0.02346 | reject |
| lambda_l2=1.0 | 0.02774 | reject (below selected by 0.00025) |

Selection rule: simplest trial within 0.0002 of the best development mean.
`lr03_n500` is uniquely best. Later stages train with 500 trees and `lr=0.03`.

Hard-negative downsampling, same 500-tree schedule:

| Scheme | Mean MAP@12 | Train rows (Aug 31 fold) |
|---|---:|---:|
| all candidates | 0.02799 | 5,350,207 |
| hard50 + rand50 | 0.01878 | 403,675 |
| hard100 + rand50 | 0.01815 | 603,675 |
| hard50 + rand150 | 0.02204 | 803,675 |

Reject. Cutting the 1K+ group to a few hundred negatives throws away the
easy-negative contrast LambdaRank was using. Keep full candidate groups.

Targeted crosses (`ALS×TT`, `TT×sources`, `sources×log pop`, repeat/recency,
rank gaps) scored **0.02606** vs **0.02799**. LightGBM already captures those
interactions. Keep the original feature set.

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
   CatBoost YetiRank                   merge / RRF
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
- Adding MiniLM preserved exact structured-query NDCG@10 at 1.0 and slightly
  increased style Recall@50 from 0.711 to 0.713, but style NDCG@10 fell from
  0.775 to 0.706 and MRR from 0.788 to 0.692. Reject it as the default under
  this synthetic benchmark; it remains available with `SEARCH_SEMANTIC=1`.
  See `artifacts/overnight/search_semantic_ablation.json`.
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
| LambdaRank truncation 12/20/30 | 0.02313 / 0.02538 / 0.02707 vs 0.02707 | Reject; default truncation is enough. |
| `rank_xendcg` | 0.02346 | Reject; LambdaRank remains the objective. |
| Faster 200-tree / lr=0.10 schedule | 0.02382 | Reject. |
| 127 leaves or looser/tighter min_child | 0.02494 / 0.02600 / 0.02639 | Reject. |
| Hard-negative downsampling | 0.01815–0.02204 vs 0.02799 all-candidates | Reject; keep full groups. |
| Explicit ranking crosses | 0.02606 vs 0.02799 baseline features | Reject; trees already interact. |
| Extra-week ranking supervision | 0.02809 vs 0.02799 | Reject; lift inside 0.0002. |
| Set-attention ListNet | crashed signal 11 on fold 1 after tiny-overfit passed | Reject; do not serve an unmeasured reranker. |

## Feature / interaction findings

The six-source feature set already includes per-source ranks, scores, source
count, repeat, recency, and popularity. Adding six explicit crosses made
development MAP worse (0.02606 vs 0.02799). LightGBM trees were already
using those relationships. Do not keep the derived columns.

## Data / supervision findings

Hard-negative downsampling reduced training rows from ~5.4M to under 1M and
hurt MAP by 0.006–0.010. Extra earlier weeks (2020-08-03 and 2020-08-10)
caches were written. Training on six weeks instead of four raised mean
MAP@12 from 0.02799 to 0.02809 and roughly doubled positives (3,675→7,568
on the Aug 31 fold). That is inside the 0.0002 band. Keep the four-week
schedule.

## Remaining bottleneck

The candidate pool already contains most recoverable positives (oracle MAP@12
≈ 0.29; candidate recall ≈ 0.28). LambdaRank promotes repeats and
multi-source popular items and buries complementary-retriever positives
(ALS-only / PMI-only / two-tower-only top-12 rates under 3%). Buried
long-tail positives and high-ranked implicit negatives look similar on the
current features. That is a **features + supervision** bottleneck, not a
missing neural ranker. Irreducible next-week purchase uncertainty is also
large: even a perfect ranker of retrieved items cannot reach oracle MAP.

## Production architecture recommendation

Until later stages overturn it, serve:

```text
Popularity + Repeat + PMI + ALS + metadata content + metadata two-tower K=50
        ↓
CatBoost YetiRank (500 iterations, lr=0.03, depth 6)
        ↓
Top 12
```

LightGBM `lr03_n500` remains the strongest gradient-boosted baseline and the
control for later stages.

Keep search as a separate query-first hybrid. Do not reuse the purchase
LambdaRank as a search ranker. Do not add GenRec, DCN, or image retrieval
without a new measured failure.

## What would change with real H&M production data

This dataset has purchases only. Real production data would change both
training and evaluation:

- Impressions and clicks would let ranking learn what was seen and skipped,
  not only what was bought.
- Carts, favorites, and returns would separate intent from fulfillment.
- Session context and real-time state would make sequential models worth
  retrying.
- Inventory and availability would make an unavailable high-score item an
  operational error instead of a silent label.
- Real query/click logs would replace the synthetic search benchmark.

Until those exist, MAP@12 on later purchases is a recovery metric, not a
causal lift or satisfaction claim.

## Best system

Status: frozen_for_holdout

```text
{
  "candidate_sources": [
    "recent_7d_pop",
    "repeat_purchase",
    "cooccurrence",
    "als",
    "content",
    "two_tower"
  ],
  "two_tower_k": 50,
  "ranker": {
    "family": "catboost_yetirank",
    "objective": "lambdarank",
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": 42,
    "two_tower_k": 50,
    "sources": [
      "recent_7d_pop",
      "repeat_purchase",
      "cooccurrence",
      "als",
      "content",
      "two_tower"
    ],
    "params_update": {},
    "selected_trial": "lr03_n500"
  },
  "reranker": null,
  "feature_set": "six_source_lambdarank_v1",
  "training_snapshots": [
    "2020-08-17",
    "2020-08-24",
    "2020-08-31",
    "2020-09-07"
  ],
  "serving_cutoff": "2020-09-15",
  "group_weighting": null,
  "negative_sampling": "all_candidates",
  "label_weighting": null
}
```

- Development baseline mean MAP@12: 0.027068315668018046
- Selected LightGBM trial: lr03_n500 (0.02799008240744645)
- Customer-holdout MAP@12: 0.030677543673145336
- Holdout customers: 8000
- Decisions: lightgbm:lr03_n500, supervision:current, negatives:all_candidates, features:baseline, weights:none, group_weights:none, ranker:catboost, reranker:none

Holdout is a customer-holdout evaluation on 2020-09-16..22, not a pristine unseen week.

## Search product

- Synthetic structured holdout: {'queries_scored': 15, 'precision_at_10': 1.0, 'recall_at_50': 1.0, 'ndcg_at_10': 1.0, 'mrr': 1.0}
- Style curated: {'queries_scored': 9, 'precision_at_10': 0.6666666666666666, 'recall_at_50': 0.7111111111111111, 'ndcg_at_10': 0.7752239300332568, 'mrr': 0.7878787878787878, 'label': 'style/token overlap on curated queries; not production search quality'}
- Search ranker: hybrid_fusion

## Git checkpoints

```text
454c534 fix: require fold-consistent lift before keeping group weights
7afe298 fix: pass long-tail rarity as LightGBM sample weights
aeede7a experiment: promote CatBoost YetiRank and reject the crashed listwise reranker
4b53f0b docs: reject extra-week supervision after a 0.00010 MAP lift
559f8f9 docs: reject explicit ranking crosses after they lost to the base features
6ede2dc docs: reject hard-negative downsampling and record the 500-tree champion
6a66f93 docs: record the ranking bottleneck and re-run search on customer switch
5882671 experiment: keep the 500-tree LightGBM schedule and add group weights
7277eb5 fix: carry the selected LightGBM trial into later overnight stages
d1cf751 docs: record the first material LightGBM lift from a slower 500-tree schedule
da82045 feat: auto-finalize after the research ladder and ship a production web image
16b6f31 docs: record rejected branches and LightGBM tune outcomes so far
bc53c24 fix: make holdout replay CatBoost/listwise and export all 8K customers
189dd59 fix: train holdout and demo slates from the frozen BEST_SYSTEM spec
387564c fix: stop search from showing zero results while a query is in flight
c67d77d feat: add search demo surfaces and synthetic style eval
81a4b7f feat: add holdout evaluation path and search benchmark helpers
c269086 feat: add hybrid search pipeline and recommendation API shell
cbbfe17 experiment: ranking research stages and frozen baseline protocol
0aa49c6 checkpoint: evaluation audit and experiment harness
c4683da checkpoint: pre-overnight recommender state
```

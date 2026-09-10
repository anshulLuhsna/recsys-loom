# RecSys Loom

Rebuilding the evolution of recommendation systems—from popularity and collaborative filtering to multimodal retrieval, learned ranking, and LLM-native recommendation—using real H&M fashion data.

## Thesis

Recommendation systems did not evolve by repeatedly replacing the previous model. They evolved by combining more kinds of evidence inside a staged system while making increasingly explicit tradeoffs around relevance, latency, freshness, constraints, exploration, and user value.

RecSys Loom rebuilds that progression from first principles. Every additional layer must solve a measured limitation of the previous system and earn its complexity through a controlled experiment.

## What this project is trying to learn

- How popularity, collaborative behavior, product content, sequences, images, and language capture different kinds of intent.
- Why candidate generation, ranking, and slate construction are separate problems.
- How recommendation quality changes when models meet catalog constraints, stale data, latency budgets, and failure conditions.
- Where LLMs genuinely improve recommendation and where deterministic retrieval or smaller learned models remain better.
- Which offline improvements survive honest temporal evaluation—and which merely make the architecture look sophisticated.

## The evolution being rebuilt

```text
Popularity
    ↓
Item-item similarity and collaborative filtering
    ↓
Feature-based retrieval and ranking
    ↓
Two-tower retrieval and approximate nearest neighbours
    ↓
Sequences, graphs, and real-time user state
    ↓
Text-image multimodal retrieval
    ↓
Learned ranking and multi-objective reranking
    ↓
LLM-assisted and generative recommendation
```

These stages are not assumed to replace one another. The final system may combine several specialized candidate sources and rank their output together.

## Core system shape

```text
Catalog and interaction data
          ↓
Candidate generation
          ↓
Pre-ranking
          ↓
Personalized ranking
          ↓
Diversity and catalog constraints
          ↓
Final slate of 12 recommendations
```

Retrieval answers whether the system can find plausible products. Ranking answers which of those products this customer should see now. Reranking decides whether the complete list is useful, available, sufficiently diverse, and safe to serve.

## Dataset

The initial dataset is from the H&M Personalized Fashion Recommendations competition:

- `articles.csv`: product metadata.
- `customers.csv`: anonymized customer metadata.
- `transactions_train.csv`: dated purchase interactions.
- Product images will be added later when the transaction-only baseline is trustworthy.

The raw dataset is kept locally and must not be committed to Git. Its competition terms continue to apply.

H&M provides purchases, not impressions, clicks, natural-language queries, explicit dislikes, inventory, or browsing sessions. Experiments and claims must respect those limits.

Personalized home recommendations are trained/evaluated using historical H&M purchase data.

Search is a hybrid retrieval/ranking demonstration built from product metadata,
lexical signals, and CLIP text-to-image similarity. MiniLM text semantics remain
optional. Because the H&M dataset contains no real search-query/click logs,
search relevance is evaluated using synthetic, structured, and small manual
benchmarks rather than production search behavior.

The optional LLM path is **GenRec-inspired**, not a reproduction of Netflix
GenRec. This repository lacks the query/search interaction labels and serving
infrastructure needed to post-train and serve a decoder model with catalog-item
scoring. Instead, the LLM may add validated soft intent and reorder only a
bounded catalog candidate set. Deterministic parsing owns explicit color,
product-type, and section interpretation. Color is a hard filter only when a
recognized product type is also present; color-only queries remain soft. BM25
always receives the raw query; returned IDs are validated; hard filters are
reapplied; and failures fall back to the existing deterministic pre-rank.
Customer history and personalization-influenced candidate order are not sent
to the LLM.

The provider path is implemented but opt-in. It is disabled when
`SEARCH_LLM=0` or credentials are absent. Live local checks with Groq's
`openai/gpt-oss-120b` verified the intended routing: a typo-heavy explicit query
(`i want blu women dres`) is grounded as blue womenswear dresses and uses
MiniLM/CLIP retrieval without paying for an LLM call; an open-ended query
(`something breezy and elegant for a beach dinner`) uses LLM intent enrichment
and bounded candidate reranking, with no invalid IDs or hard-filter violations.
Exact catalog wording remains eligible for BM25-first retrieval. These are
behavior checks, not a relevance-quality benchmark. The evaluation harness
freezes each variant's rankings before producing shuffled, image-visible
judging sheets, so future human judgments can remain blind to which system
returned each item.

## First prediction task

Given everything known about a customer before a cutoff, rank 12 articles that the customer is likely to purchase during the next seven days.

The first local validation split is:

```text
Training:   2018-09-20 through 2020-09-15
Validation: 2020-09-16 through 2020-09-22
```

No validation-week information may influence features, candidates, popularity counts, or model training.

## Experiment ladder

Each stage should answer one question before the next one begins:

1. **Popularity:** How far can recency and global demand go? **Done.**
2. **Repeat purchase:** How much signal exists in a customer's own history? **Done.**
3. **Item-item retrieval:** Do products purchased together produce better candidates? **Done.**
4. **Collaborative filtering:** Does shared customer behaviour reveal affinity beyond popularity? **Done.**
5. **Content retrieval:** Can product metadata improve sparse-user and cold-item recommendations? **Done.**
6. **Two-tower learned retrieval:** Can feature-built user/item representations recover new purchases? **Done.**
7. **Sequential modelling:** Does the order and recency of purchases improve immediate prediction? *Deferred.*
8. **Richer item representations:** Do image or text semantics improve retrieval?
   **Image-only retrieval and text-augmented two-tower tested; neither retained.**
   Joint text-image retrieval remains deferred.
9. **Learned ranking:** Can the available signals be combined better than fixed rules? **Done: CatBoost YetiRank retained.** See [`OVERNIGHT_RESULTS.md`](OVERNIGHT_RESULTS.md).
10. **LLM layer:** Does language understanding add measurable value beyond embeddings and conventional rankers? **Live path verified; blinded human relevance judgments pending.**
11. **Constrained reranking:** Can relevance survive diversity, availability, freshness, and latency requirements? *Deferred: the dataset has no inventory or freshness truth.*

Only one meaningful variable should change between adjacent experiments whenever possible.

## Retrieval results (5K sample, validation week 2020-09-16 to 2020-09-22)

Five candidate sources retained. Three redundant popularity variants dropped.

| Source | What it does | R@100 | R@500 | Unique hits @100 | Catalog coverage |
|---|---|---|---|---|---|
| recent_7d_pop | Recommends trending items from the last 7 days | 0.120 | 0.320 | 30% | 0.5% |
| repeat_purchase | Re-suggests items the customer bought before | 0.054 | 0.054 | 58% | 48.4% |
| cooccurrence | Items co-purchased with this customer's items (PMI) | 0.042 | 0.096 | 57% | 15.3% |
| content | Multi-hot FAISS vector similarity on categorical features | 0.020 | 0.050 | 52% | 68.7% |
| als | Collaborative filtering via ALS (64 factors, 15 iterations) | 0.052 | 0.114 | 23% | 18.2% |

No single source dominates. Each finds purchases the others miss. Content had the original SQL cross-join approach rewritten to FAISS vector retrieval after out-of-memory failures on the full dataset.

## Two-tower learned retrieval (5K development sample)

The sixth source constructs 64-dimensional user and article embeddings from
features rather than customer/article ID embeddings. The user tower aggregates
up to 50 recency-weighted purchases plus customer statistics. The item tower
uses eight learned categorical embeddings. Training uses four leakage-safe
weekly snapshots with in-batch and sampled-softmax negatives.

| Metric | Two-tower | Existing 5-source union | Union + two-tower | Marginal |
|---|---:|---:|---:|---:|
| Recall@100 | 0.0290 | 0.2117 | 0.2241 | +0.0124 |
| Recall@300 | 0.0621 | 0.3502 | 0.3716 | +0.0214 |
| Recall@500 | 0.0882 | 0.4375 | 0.4636 | +0.0261 |

The source is weaker standalone than ALS but highly complementary: candidate
Jaccard overlap with ALS is only 1.1% at 100 and 3.0% at 500. Candidate-pool
oracle MAP@12 rises from 0.4519 to 0.4756.

FAISS IVF search runs in an isolated process because FAISS and PyTorch load
conflicting OpenMP runtimes on this macOS environment. ANN and exact Top-20
have the same cutoff score within `1e-4` for every checked query; exact article
ID overlap is lower when many metadata-identical items tie.

## Initial five-source vs six-source ranking

Two-tower models and candidates are rebuilt independently for every ranking
snapshot from earlier weekly data. The default five-source path remains
unchanged.

On the 5K final validation population, adding two-tower raises candidate
recall from 0.4375 to 0.4636, candidate hit rate from 0.6652 to 0.6850, and
oracle MAP@12 from 0.4519 to 0.4756. The average union grows from 1,655 to
2,051 candidates.

The initial LambdaRank treatment was not consistently better:

| Validation cutoff | 5-source MAP@12 | 6-source MAP@12 | Delta |
|---|---:|---:|---:|
| 2020-08-31 (2K) | 0.0250 | 0.0220 | -0.0030 |
| 2020-09-07 (2K) | 0.0254 | 0.0256 | +0.0001 |
| 2020-09-15 (5K) | 0.0250 | 0.0237 | -0.0013 |

This comparison later exposed a candidate-depth mismatch: historical ranking
snapshots contained 300 two-tower candidates per customer, while the final 5K
cache contained 500. The retrieval and oracle measurements remain valid, but
the `0.0237` ranking result must not be treated as a controlled K=300 result.
The fixed-budget sweep below supersedes it for selecting the retrieval blend.

Two-tower adds most marginal recall for zero-history customers (+0.0405) and
customers with 6–10 historical purchases (+0.0322). At K=500 its candidate
Jaccard overlap with ALS is 0.0297 and relevant-hit Jaccard is 0.1288. Its
marginal recall is larger for long-tail than popular future items (0.0355 vs
0.0173), and for newer than established items (0.0311 vs 0.0207).

## Two-tower candidate-budget sweep

The controlled sweep keeps the existing source budgets and 41-feature
LambdaRank representation fixed, varies only two-tower K, and retrains the
ranker for every treatment. Two earlier 2K folds select the budget; the 5K
final week is not used for selection. Latency below is LambdaRank scoring time,
not ANN retrieval time.

| TT K | Avg candidates | Candidate recall | Oracle MAP@12 | LambdaRank MAP@12 | HR@12 | Scoring ms/customer |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,654.6 | 0.4375 | 0.4519 | 0.0263 | 0.1146 | 3.81 |
| 25 | 1,673.9 | 0.4407 | 0.4549 | 0.0277 | 0.1124 | 3.94 |
| 50 | 1,693.0 | 0.4424 | 0.4562 | 0.0269 | 0.1156 | 3.90 |
| 100 | 1,732.0 | 0.4452 | 0.4594 | 0.0279 | 0.1162 | 3.95 |
| 200 | 1,809.9 | 0.4500 | 0.4634 | 0.0270 | 0.1176 | 4.12 |
| 300 | 1,888.8 | 0.4541 | 0.4672 | 0.0260 | 0.1090 | 4.09 |

The earlier-fold MAP@12 results were:

| TT K | 2020-08-31 | 2020-09-07 | Mean |
|---:|---:|---:|---:|
| 0 | 0.0264 | 0.0254 | 0.0259 |
| 25 | 0.0246 | 0.0274 | 0.0260 |
| 50 | 0.0264 | 0.0278 | 0.0271 |
| 100 | 0.0261 | 0.0271 | 0.0266 |
| 200 | 0.0248 | 0.0268 | 0.0258 |
| 300 | 0.0264 | 0.0263 | 0.0263 |

The frozen selection rule chooses the smallest K within 0.0002 MAP@12 of the
best earlier-fold mean, selecting K=50. On the final week it improves MAP@12
from 0.0263 to 0.0269 while adding 38 candidates per customer and 0.48
percentage points of candidate recall. Marginal retrieval efficiency declines
as K grows: relevant hits per 10K added candidates fall from 5.18 at K=25 to
2.23 at K=300. K=50 is therefore the new budgeted two-tower blend.

An earlier-fold history policy selected `{0: 100, 1-2: 100, 3-5: 200,
6-10: 300, 11-20: 100, 20+: 50}`. It achieved only 0.0258 MAP@12 on the final
week, below both K=0 and global K=50. Segment-specific choices were unstable
on the small selection buckets, so history-conditioned routing is rejected
for now rather than tuned on the final week.

Reproduce:

```bash
python scripts/run_two_tower_budget_sweep.py 2000 5000
```

## DCN V2 with temporal two-tower embeddings (2K matched sample)

The neural ablation adds each snapshot's 64-dimensional two-tower user and
item vectors without duplicating them for every candidate row. Model selection
uses the 2020-09-07 snapshot; models are then refit on all four earlier
snapshots for the selected epoch count and evaluated once on 2020-09-15.

| Ranker | MAP@12 | Recall@12 | Hit Rate@12 |
|---|---:|---:|---:|
| Six-source LambdaRank | 0.0275 | 0.0510 | 0.1275 |
| MLP + two-tower embeddings | 0.0227 | 0.0402 | 0.1075 |
| DCN V2 + two-tower embeddings | 0.0148 | 0.0291 | 0.0815 |
| Candidate oracle | 0.3388 | 0.3222 | 0.5595 |

The MLP has 20,329 parameters and selects epoch 1. DCN V2 has 115,158
parameters and selects epoch 5. DCN training loss continues falling while
temporal-validation loss rises after epoch 4–5, showing overfitting rather
than a broken optimizer. Snapshot embedding lookup was checked against stored
two-tower scores on 10K pairs per cutoff; maximum absolute error was below
`1.2e-7`.

Learned vectors substantially narrow the MLP gap, but neither neural ranker
beats LambdaRank. DCN also underperforms the simpler MLP, so explicit cross
layers are not adding useful generalizable interactions at this data scale.
LambdaRank remains the ranking baseline; representation quality can be
revisited later without further DCN tuning now.

## Image-only retrieval experiment

The image experiment encodes 105,100 available article images with the frozen
`facebook/dinov2-small` model. Each article receives a normalized
384-dimensional vector. For each temporal cutoff, a customer visual profile is
the normalized, 45-day recency-weighted mean of up to 50 previous purchases.
Previously purchased articles are removed from image candidates. No text,
metadata, target-week transactions, or neural-ranker changes enter this
experiment.

The final 5K snapshot has visual profiles for 92.1% of customers. FAISS IVF
search takes approximately 0.74 ms per profile including subprocess startup
and index construction in this offline measurement. ANN Top-20 overlap with
exact search is at least 99.9% on the checked queries.

Image retrieval is weak for next-week purchase prediction:

| Image K | Standalone recall | Marginal recall over TT-K50 baseline | Oracle gain |
|---:|---:|---:|---:|
| 50 | 0.0036 | +0.0006 | +0.0007 |
| 100 | 0.0060 | +0.0013 | +0.0014 |
| 300 | 0.0143 | +0.0038 | +0.0040 |

The LambdaRank budget sweep uses the same earlier-fold selection protocol as
the two-tower experiment:

| Image K | 2020-08-31 MAP@12 | 2020-09-07 MAP@12 | Mean |
|---:|---:|---:|---:|
| 0 | 0.0264 | 0.0278 | **0.0271** |
| 25 | 0.0248 | 0.0272 | 0.0260 |
| 50 | 0.0243 | 0.0281 | 0.0262 |
| 100 | 0.0257 | 0.0263 | 0.0260 |
| 200 | 0.0268 | 0.0252 | 0.0260 |
| 300 | 0.0253 | 0.0254 | 0.0254 |

The frozen choice is therefore `K_image=0`. On the final week, K=0 scores
0.0278 MAP@12; K=50 scores 0.0275 and K=300 falls to 0.0244. A history-based
image policy also loses at 0.0270. Image candidates add too few unique relevant
items for their volume, so image-only DINOv2 retrieval is not retained in the
production-shaped baseline.

This result rejects the current mechanism, not all visual information. A
generic image encoder plus one averaged purchase-history vector may represent
visual similarity without representing what the customer will purchase next.
Any later visual experiment must introduce a distinct hypothesis, such as
fashion-specific contrastive training or a controlled text-image model, rather
than tuning this failed budget sweep.

Reproduce:

```bash
python scripts/encode_image_embeddings.py 64
python scripts/run_image_retrieval.py 2000 5000
python scripts/run_image_budget_sweep.py 2000 5000
```

## Text-augmented two-tower experiment

This experiment tests semantic text inside the purchase-supervised item tower,
not as a standalone text nearest-neighbour source. The article audit found that
`detail_desc` is populated for 99.61% of articles and averages 142 characters
(23.9 words); `prod_name` is populated for 100%. The encoder input is:

```text
Product: {prod_name}. Description: {detail_desc}
```

The eleven categorical name fields are excluded from this string because the
metadata item tower already embeds their categorical values. Frozen
`sentence-transformers/all-MiniLM-L6-v2` produces an aligned, normalized
384-dimensional vector for all 105,542 articles. The cache occupies 154.7 MiB.

The controlled ablation changes only the item encoder. A learned 384-to-64
text projection is fused with the existing categorical representation, and the
final item vector remains 64-dimensional. The user tower still uses the same
customer inputs and recency-weighted history aggregation; historical products
pass through the shared augmented item encoder so user and candidate vectors
remain in the same space. The metadata model has 39,836 parameters and the
text model has 72,796.

Text substantially improves the two-tower retriever on the final 5K population:

| Metric | TT metadata | TT + text | Delta |
|---|---:|---:|---:|
| Recall@50 | 0.0137 | 0.0308 | +0.0171 |
| Recall@100 | 0.0237 | 0.0515 | +0.0278 |
| Recall@300 | 0.0511 | 0.1079 | +0.0568 |
| Recall@500 | 0.0733 | 0.1479 | +0.0746 |
| Hit Rate@500 | 0.1680 | 0.2926 | +0.1246 |
| Catalog coverage@500 | 0.4309 | 0.2533 | -0.1776 |

Candidate neighborhoods genuinely change: metadata versus text candidate
Jaccard is 0.0354 at K=50 and 0.0960 at K=500; relevant-hit Jaccard at K=500
is 0.2434. At the frozen initial K=50, replacing metadata TT with text TT
improves final union recall from 0.4414 to 0.4457 and oracle MAP@12 from 0.4558
to 0.4590 while adding fewer unique candidates after deduplication. All four
earlier folds also show a positive union-recall delta.

Training diagnostics pass. Both models overfit the tiny fixture, all final user
and item embedding norms equal one, and the final positive-minus-random
similarity margin rises from 0.1418 to 0.2280 with text. Final ANN search takes
approximately 0.295 ms per customer for metadata and 0.315 ms for text,
including offline subprocess overhead. Epoch-level train and temporal
validation losses are stored in the detailed report.

The first downstream result is negative: at K=50, LambdaRank MAP@12 falls from
0.02807 to 0.02644. This is outcome C—retrieval and oracle improve, but ranking
does not. The predeclared budget sweep therefore selects K only on the
2020-08-31 and 2020-09-07 folds:

| TT+text K | Earlier-fold mean MAP@12 |
|---:|---:|
| 0 | 0.02593 |
| 25 | 0.02579 |
| 50 | 0.02499 |
| 100 | 0.02680 |
| 200 | **0.02740** |
| 300 | 0.02560 |

The frozen choice is `K_text=200`; the final week is then evaluated once:

| Final metric | Metadata TT K=50 | Text TT K=200 | Delta |
|---|---:|---:|---:|
| Union candidate recall | 0.4414 | 0.4636 | +0.0222 |
| Candidate Hit Rate | 0.6694 | 0.6864 | +0.0170 |
| Oracle MAP@12 | 0.4558 | 0.4755 | +0.0197 |
| LambdaRank MAP@12 | **0.02807** | 0.02743 | -0.00063 |
| Recall@12 | **0.04323** | 0.04247 | -0.00076 |
| Hit Rate@12 | **0.1126** | 0.1102 | -0.0024 |
| Average candidates | 1693.4 | 1804.8 | +111.4 |

Text helps standalone retrieval most for popular, newer, richly described, and
metadata-ambiguous relevant products. It helps less for very low-history
products and loses on some smaller groups such as socks/tights. The semantic
signal is real, but the current LambdaRank pipeline cannot convert its added
candidates into a better top 12. Keep metadata-only TT at K=50 in the baseline
and do not retain text fusion yet.

### Dense text compatibility on the unchanged pool

A follow-up keeps the baseline candidate pool exactly unchanged—five existing
sources plus metadata TT K=50—and scores every candidate with the temporally
matched text-trained two-tower. No text candidates are added. A dense
metadata-TT score is included as a control.

| Ranker features | Earlier-fold mean MAP@12 | Final MAP@12 | Final Recall@12 | Final Hit Rate@12 |
|---|---:|---:|---:|---:|
| Existing LambdaRank | 0.02586 | **0.02807** | 0.04323 | 0.1126 |
| + dense metadata TT score/rank | 0.02511 | 0.02770 | 0.04419 | 0.1156 |
| + dense text TT score | 0.02645 | 0.02715 | **0.04527** | **0.1170** |
| + dense text TT score/rank | **0.02696** | 0.02649 | 0.04489 | 0.1162 |

Text scores separate candidate positives from negatives more strongly than
metadata scores on every snapshot, and they improve both earlier-fold mean
MAP@12 and final Recall/Hit Rate. However, the final MAP@12 result is worse and
the direction is not temporally consistent. The improvement in hit count comes
with worse ordering near the top of the 12-item slate, so the dense text score
is not retained.

The expanded-pool diagnosis explains the earlier oracle/ranker gap. Text K=200
introduces 381 relevant customer-item pairs absent from the metadata-TT K=50
pool, but their median LambdaRank position is 501 and only 3.15% reach the top
12. Every one is supported by text TT alone (`source_count=1`); already-retrieved
relevant items average 1.51 sources and 70.1% have recent-popularity support.
The new text positives are also much less popular: median 30-day purchases are
89 versus 402 for already-retrieved relevant items. LambdaRank therefore buries
the exact single-source, lower-popularity items that semantic retrieval uniquely
finds. This is useful diagnosis, but neither candidate expansion nor dense
compatibility scoring produces a final MAP gain, so text work stops here.

Detailed artifacts:

```text
artifacts/text_retrieval/text_field_audit.json
artifacts/text_retrieval/text_embedding_metadata.json
artifacts/text_retrieval/two_tower_ablation/two_tower_text_ablation_5000_2000_5000.json
artifacts/text_retrieval/two_tower_text_ranking_2000_5000.json
artifacts/text_retrieval/two_tower_text_budget_selection_2000_5000.json
artifacts/text_retrieval/two_tower_dense_score_ranking_2000_5000.json
artifacts/text_retrieval/text_expansion_rank_diagnostic_2000_5000.json
```

Reproduce:

```bash
python scripts/audit_article_text.py
python scripts/encode_article_text.py 256 8192
python scripts/run_two_tower_text_ablation.py 5000 2000 5000
python scripts/run_two_tower_text_ranking.py 2000 5000
python scripts/run_two_tower_text_budget_sweep.py 2000 5000
python scripts/export_two_tower_dense_scores.py 2000 5000
python scripts/run_two_tower_dense_score_ranking.py 2000 5000
python scripts/analyze_text_expansion_ranks.py 2000 5000
```

## Ranking results (500 customer sample)

LightGBM LambdaRank ranker trained with temporal snapshots:
- Training: history through 2020-09-07, predict 2020-09-08 to 2020-09-14
- Validation: history through 2020-09-15, predict 2020-09-16 to 2020-09-22

All candidate sources are regenerated per snapshot from scratch (separate ALS model, popularity counts, co-occurrence pairs, content vectors, repeat-purchase histories). No information from the target period leaks into any retrieval model or feature.

| Metric | LambdaRank | Popularity baseline | Lift |
|---|---|---|---|
| MAP@12 | 0.0184 | 0.0103 | +79% |
| Recall@12 | 0.0378 | 0.0288 | +31% |
| Hit Rate@12 | 0.0920 | 0.0840 | +10% |

37 features per (customer, candidate) pair: source flags/ranks/scores, source count, category affinity (6 attributes), recency and frequency of item/type/group purchases, item popularity stats (7d/30d counts, growth, unique buyers).

Top feature groups by gain: item popularity stats, category affinity, ALS scores, co-occurrence ranks, purchase recency.

Reproduce:
```bash
python scripts/run_ranking.py 500
```

### Deferred retrieval techniques

These belong to later experiment-ladder stages and will be revisited after the ranking model establishes a baseline MAP@12:

- **Sequential modelling (step 6):** Purchase-order patterns. Limited by lack of session/click data; only purchase dates are available.
- **Text-image multimodal retrieval:** Deferred. The image-only DINOv2 source
  was tested and rejected; a later experiment needs a different representation
  hypothesis rather than another K sweep.
- **Graph-based retrieval:** Random walks or GNN on user-item bipartite graph (PinSage-style). Revisit if co-occurrence PMI proves too sparse.

## Evaluation

The primary ranking metric is MAP@12, matching the H&M competition. It is accompanied by:

- Recall@12 for candidate and final-list recovery.
- Hit Rate@12 for per-customer usefulness.
- Catalog coverage and popularity concentration.
- Performance for cold users, cold items, and long-tail products.
- Constraint-violation rate once operational fields exist.
- Inference latency and index freshness once the system is served.

Models are evaluated on multiple chronological weekly splits. Random transaction splits are not valid because they leak future behaviour into the past.

Offline purchase prediction does not establish causal product value. It shows that the system recovered products purchased later, not that presenting those recommendations caused the purchase.

## Lessons guiding the design

1. A recommender is a production system, not one model.
2. New methods usually contribute another signal rather than making older methods useless.
3. Retrieval quality and ranking quality must be measured separately.
4. Labels, negative sampling, temporal splits, and candidate generation can matter more than model novelty.
5. Long-term preference, changing interests, and immediate intent operate on different timescales.
6. LLMs are useful for intent understanding, semantic representations, supervision, and explanation—but they do not remove the need for catalog grounding, constraints, or efficient serving.
7. An unavailable or stale recommendation is incorrect even when its relevance score is high.
8. Offline accuracy is not the same as user satisfaction, causal lift, diversity, or marketplace health.

## Working agreement

Anshul owns scope, architecture, and experiment decisions. AI tools implement directly when asked. The purpose is to build a working system backed by controlled evidence, not to produce a repository that merely looks sophisticated.

## Arbityr decision reviews

Arbityr is used as a recurring decision critic, not as an implementation autopilot. Its job is to challenge whether a proposed layer solves a real measured constraint and whether the added complexity is justified.

The project-scoped skill can be installed with:

```bash
gh skill install anshulLuhsna/arbityr-skill arbityr --agent codex --scope project
```

After installation, verify that these files exist:

```text
.agents/skills/arbityr/SKILL.md
.agents/skills/arbityr/resources/cursor-rule.mdc
```

Run an Arbityr review at these checkpoints:

1. Before introducing a new candidate source or model family.
2. After a baseline or experiment produces results.
3. Before changing the evaluation protocol or success metric.
4. Before adding expensive infrastructure, online serving, or synthetic operational data.
5. Before making a public technical claim in the final article.

Each review should be grounded in repository evidence and answer:

- What measured limitation are we addressing?
- What is the smallest experiment that can test the proposed idea?
- What evidence would justify keeping it?
- What complexity, latency, cost, or reliability tradeoff does it introduce?
- What result would make us reject or remove it?

Record the decision and its evidence in the repository. Arbityr advises and challenges; Anshul owns the final product and architecture decision.

## Immediate milestone

Build the smallest complete recommendation loop by hand:

```text
Raw transactions
      ↓
Data audit
      ↓
Leakage-safe temporal split
      ↓
Recent-popularity baseline
      ↓
Hand-tested MAP@12 evaluator
      ↓
Reproducible metrics and predictions
```

Images, collaborative filtering, learned rankers, and LLM components wait until this loop is trustworthy.

## Data audit

The first read-only audit of the local H&M CSV files is reproducible with DuckDB 1.5.5:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/data_audit.py
```

The generated measurements are recorded in [`reports/data_audit.md`](reports/data_audit.md). Their modeling implications and unresolved decisions are recorded separately in [`reports/data_audit_findings.md`](reports/data_audit_findings.md).

Small machine-readable query outputs are written under `artifacts/data_audit/` and intentionally ignored by Git. The raw CSV files are read directly and are never modified.

The first all-history global-popularity result is recorded in [`reports/popularity_baseline.md`](reports/popularity_baseline.md). Reproduce it with:

```bash
python -m unittest discover -s tests -v
python scripts/run_popularity_baseline.py
```

The controlled seven-day recency comparison is recorded in [`reports/recent_popularity_7d.md`](reports/recent_popularity_7d.md). Reproduce it with:

```bash
python scripts/run_recent_popularity_7d.py
```

The stability check across eight consecutive chronological weeks is recorded in [`reports/popularity_backtest_8w.md`](reports/popularity_backtest_8w.md). Reproduce it with:

```bash
python scripts/run_popularity_backtest.py
```

## Running the recommendation and search app

The overnight evaluation protocol, ranking failure analysis, and current
`BEST_SYSTEM` live in [`OVERNIGHT_RESULTS.md`](OVERNIGHT_RESULTS.md) and
[`docs/evaluation_ledger.md`](docs/evaluation_ledger.md). Development
selection uses the 2020-08-31 and 2020-09-07 2K folds. The official final
number is a reserved 8,000-customer holdout, not an unseen calendar week.

Frozen `BEST_SYSTEM`: six-source CatBoost YetiRank (500 iterations,
`lr=0.03`). Development mean MAP@12 0.02987 vs LightGBM 0.02799 vs 300-tree
baseline 0.02707. Official customer-holdout MAP@12 is **0.03068** on 8,000
unused 2020-09-16..22 buyers. That is not an unseen calendar week. Copy
`env.example` to `.env` if you need to change ports.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/export_demo_customers.py
# After BEST_SYSTEM is frozen, also:
# python scripts/export_demo_recommendations.py
# One-time local CLIP artifact build:
# python scripts/encode_visual_search_images.py 64
SEARCH_SEMANTIC=0 SEARCH_VISUAL=1 SEARCH_LLM=0 uvicorn services.recommender.app:app --reload --port 8000
```

In another shell:

```bash
cd apps/web
npm install
npm run dev
```

Open `http://127.0.0.1:3000` for personalized For You recommendations and
`http://127.0.0.1:3000/search` for hybrid catalog search.

Docker:

```bash
docker compose up --build
```

The API image serves FastAPI. The web image is a production Next.js build.
Mount local `artifacts/`, `articles.csv`, and optional `images/` into the API
container. Set `SEARCH_SEMANTIC=0` to skip MiniLM. Set `SEARCH_VISUAL=1` only
when `artifacts/visual_search/` contains the encoded CLIP matrix. LLM search is
off by default. To opt in, set `SEARCH_LLM=1` and configure the
OpenAI-compatible settings shown in `env.example`; `GROQ_API_KEY` is accepted
as a fallback credential and secrets must remain outside Git. External calls
are bounded per process by `SEARCH_LLM_MAX_CALLS_PER_MINUTE` (default `30`);
quota exhaustion uses deterministic fallback.

The recommendation endpoint prefers live CatBoost inference from the exported
`artifacts/serving/` bundle. The current bundle contains the frozen
`BEST_SYSTEM` model and 2,678,481 candidate-feature rows for 2,000 customers at
the 2020-09-15 serving cutoff. The process loads and caches the model and
article index lazily, and loads each customer's compressed feature block on
demand into a 128-customer cache. Each request scores the already-exported
candidate rows; it does not regenerate candidates, retrain, or refit CatBoost.
If the live bundle is absent or invalid, the five curated demo customers fall
back to precomputed Top-12 slates in
`artifacts/overnight/demo_recommendations.json`; other unsupported IDs return
404.

In-process verification on 2026-09-09 measured 482.8 ms end-to-end for the
first recommendation request and 1.7 ms for a repeated request for the same
customer. The response's ranker-path measurement was 25.2 ms cold and 0.6 ms
warm; the larger first end-to-end number includes initial catalog loading.
The search engine builds its in-memory BM25 and structured indexes lazily on
the first search request and then reuses them. When `SEARCH_SEMANTIC=1`, the same lazy initialization also
loads the MiniLM query encoder and frozen article-text embeddings. MiniLM is
off by default because it did not improve the lexical/token-overlap synthetic
benchmark. That benchmark is circular and favors BM25, so real semantic-search
value remains inconclusive. With `SEARCH_VISUAL=1`, the same lazy
initialization loads the frozen CLIP image matrix and text encoder.

Raw H&M CSVs and the full image archive stay local and are not shipped as Git
artifacts. The public demo should use a precomputed demo-customer subset plus
catalog thumbnails, not tens of gigabytes of raw files.

Reproduce overnight ranking experiments from the development protocol:

```bash
python scripts/audit_evaluation_integrity.py
python scripts/run_overnight_baseline.py
python scripts/analyze_ranking_failures.py
```

## Vercel frontend and EC2 backend lab

This is a low-traffic portfolio/interview stack. The browser talks to Next.js
on Vercel and to `https://api.<your-domain>` on one EC2 VM. Caddy owns ports
80/443. Recommendation and search run as separate Compose services; search
falls back to unpersonalized results if recommendation is slow or down.

Do not put raw H&M transactions, `.env`, or training caches on the VM. Build
a synthetic bundle first:

```bash
python scripts/create_interview_lab_bundle.py /tmp/recsys-loom-lab-bundle \
  --archive /tmp/recsys-loom-lab-bundle.tar.gz
```

Branching is protected `main`, short-lived feature PRs, and hotfixes from the
deployed commit. Frontend production is Vercel Git integration: project root
`apps/web`, production branch `main`,
`NEXT_PUBLIC_API_URL=https://api.<your-domain>`. Preview deployments need a
matching CORS origin; do not allow `*`.

Backend CD is GitHub Actions OIDC to AWS. After `infra/bootstrap` and
`infra/live` are applied, set repository variables
`AWS_REGION`, `AWS_DEPLOY_ROLE_ARN`, `ECR_RECOMMENDATION_REPOSITORY`,
`ECR_SEARCH_REPOSITORY`, `DEPLOYMENT_CONFIG_BUCKET`, `ALLOWED_ORIGINS`,
`BACKEND_BASE_URL`, `SSM_INSTANCE_ID`, plus the serving-bundle URI/version.
Protect `main` and require the `production` environment on
`.github/workflows/deploy-backend.yml`.

Terraform apply order:

```bash
cd infra/bootstrap
# copy terraform.tfvars.example, then:
terraform init
terraform apply

cd ../live
# copy backend.hcl.example, tenant.tfvars.example, environment.tfvars.example
terraform init -backend-config=backend.hcl
terraform plan -var-file=tenant.tfvars -var-file=environment.tfvars
terraform apply -var-file=tenant.tfvars -var-file=environment.tfvars
```

SSH interview drills stay on a disposable tenant with
`enable_lab_fault_volume = true` and a candidate `/32` SSH CIDR. On the VM:

```bash
LAB_MODE=1 LAB_CONFIRM=SYNTHETIC_ONLY /opt/recsys-loom/app/lab/enable.sh
LAB_MODE=1 /opt/recsys-loom/app/lab/service-stop.sh search-api
LAB_MODE=1 /opt/recsys-loom/app/lab/cleanup.sh --disable
```

Never fill `/`, change SSH/firewall, or run unbounded load. The interviewer
rollback is `cleanup.sh` plus the previous image digest recorded by
`ops/deploy.sh`.

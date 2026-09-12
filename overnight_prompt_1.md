You are taking ownership of the H&M recommender project for an overnight autonomous research run.

Your job is NOT merely to implement one requested model.

Your job is to act as the **ML research lead + recommender systems engineer** for the repository, diagnose the current bottleneck, run a disciplined experiment ladder, automatically prune weak branches, repair failures when reasonable, and leave the repository in a clean, reproducible, documented state with the strongest scientifically defensible system you can produce.

I have the whole night available. Compute usage is not a major concern. Correctness, experimental integrity, reproducibility, and understanding WHY something works are more important than saving a little compute.

The goal is:

> Maximize final recommendation quality, especially MAP@12, while ensuring every improvement comes from controlled experiments that let us understand why it worked.

---

# 0. FIRST RESPONSE: GIVE ME CHOICES, THEN PROCEED

Before touching code, briefly inspect the repository enough to understand the current state, then show me three possible overnight strategies:

### Option A — Conservative

Only debug and optimize the existing LambdaRank system.

### Option B — Ranking-focused

Do ranking failure analysis, increase supervision/data, improve LambdaRank, and test one alternative ranker.

### Option C — Full overnight research program

Run a staged experiment tree:

1. evaluation-integrity audit;
2. ranking failure analysis;
3. scale training data/supervision;
4. targeted feature/supervision experiments;
5. serious LambdaRank tuning;
6. alternative tree ranking objective/model;
7. listwise/set-aware reranking experiment;
8. optional GenRec-inspired final reranker if justified;
9. full-scale evaluation;
10. final documentation.

Recommend **Option C**.

Because this is an overnight autonomous run and I may not respond, do NOT block waiting for confirmation.

After showing the choices, state:

> "I recommend Option C and will proceed with it unless interrupted."

Then continue automatically.

---

# 1. CURRENT PROJECT STATE

Read the current README, experiment reports, artifacts, and code rather than assuming the following numbers are perfectly current.

The project is an H&M next-purchase recommendation system.

Goal:

```text
~105K articles
      ↓
candidate generation / retrieval
      ↓
~1K-ish candidate pool
      ↓
ranking
      ↓
Top 12 recommendations
```

Primary final metric:

```text
MAP@12
```

Retrieval is mainly judged by:

```text
Recall@K
HitRate@K
marginal union recall
candidate count
latency
oracle MAP@12
```

The current retrieval stack includes approximately:

* popularity
* repeat purchase
* PMI / item-item co-purchase
* ALS
* metadata content retrieval
* metadata two-tower, globally limited to K=50

Important completed findings:

### LambdaRank

Multi-week LightGBM LambdaRank is currently the strongest ranking family.

Increasing positives from approximately:

```text
446 → 7,177
```

caused a major gain:

```text
MAP@12 ~0.015 → ~0.0256
```

This was much more valuable than changing ranker architecture.

### DCN / MLP

Neural rankers have been tested and debugged.

Implementation checks passed:

* normalization
* gradient flow
* BPR sign
* tiny overfit sanity test
* embedding alignment
* hard-negative training

But neural rankers generalized poorly.

Representative controlled result:

```text
LambdaRank            ~0.0275
MLP + learned embeds  ~0.0227
DCN V2 + embeds       ~0.0148
```

DCN overfits and is deferred.

Do NOT spend the night blindly tuning DCN.

### Two-tower

Two-tower is useful as a complementary retriever despite weak standalone recall.

It has very low ALS overlap and finds some different items, particularly long-tail/newer items.

A controlled K sweep selected:

```text
K_TT = 50
```

using earlier temporal folds.

On the untouched matched test at that point:

```text
No TT       MAP@12 = 0.02633
TT K=50     MAP@12 = 0.02688
```

K=100 happened to score higher on that final week but was correctly rejected because it would constitute tuning on final validation.

History-based TT routing was also rejected.

### Images

DINOv2 image-only retrieval was tested and rejected.

It added weak marginal recall and hurt final MAP.

Do not revisit generic averaged image retrieval tonight unless a later result provides a very strong reason.

### Text

Text added real semantic retrieval signal.

For example:

```text
TT Recall@500:
0.0733 → 0.1479
```

and increased union/oracle recall.

But:

* expanding the candidate pool with text hurt final MAP;
* using the dense text compatibility score on the unchanged candidate pool did not consistently improve MAP;
* new text positives tended to be single-source, lower-popularity items;
* their median LambdaRank position was around 501;
* only roughly 3% reached Top 12.

Text work is currently frozen.

Do not restart text experiments tonight unless ranking-failure analysis directly justifies a narrowly targeted experiment.

---

# 2. CRITICAL: AUDIT EVALUATION INTEGRITY FIRST

We have already inspected several previously called "final" weeks/results.

Therefore, do NOT blindly continue treating a previously viewed final week as a pristine holdout.

Before experiments:

1. Inspect README, artifacts and scripts.
2. Build an **evaluation ledger** describing:

   * every temporal window previously used for training;
   * every temporal window previously used for hyperparameter/model selection;
   * every temporal window whose final metrics have already been inspected;
   * customer subsets used in 500 / 1K / 2K / 5K / full experiments.

Determine whether any genuinely untouched temporal window remains.

### Preferred final evaluation protocol

If an untouched temporal window exists:

* reserve it;
* NEVER inspect it until final model selection is complete.

If no untouched temporal window remains:

* do NOT pretend otherwise.

Instead, establish the cleanest defensible holdout possible.

Preferred fallback:

> Use a disjoint customer cohort from the full ~69K population whose target-week labels were not previously used for experiment selection.

Use earlier temporal snapshots and previously used customer subsets for development/model selection.

Then freeze the architecture/hyperparameters.

Evaluate once on the new customer holdout.

Clearly document that this is a **customer-holdout final evaluation**, not a pristine unseen temporal holdout.

If even that is impossible, use nested temporal backtesting and explicitly call results retrospective/backtest results rather than claiming an untouched final test.

Experimental honesty is mandatory.

---

# 3. GIT / CHECKPOINT POLICY

I explicitly want periodic commits.

Before modifying anything:

1. inspect `git status`;
2. do not discard existing work;
3. make sure datasets, caches, images, embeddings, secrets, API keys, temp files, DuckDB spills, virtual environments, checkpoints and large model artifacts are Git-ignored;
4. create a safe pre-overnight checkpoint.

If appropriate:

```text
checkpoint: pre-overnight recommender state
```

Create a branch such as:

```text
overnight-ranking-research
```

or use the project's existing branch convention.

Commit after meaningful milestones, NOT after every tiny edit.

Suggested checkpoints:

```text
checkpoint: evaluation audit and experiment harness
experiment: ranking failure analysis
experiment: scaled ranking supervision
experiment: lambdarank optimization
experiment: alternative tree ranker
experiment: listwise reranker
checkpoint: selected overnight architecture
docs: overnight experiment report
```

Commit:

* source code;
* configs;
* small JSON/CSV result summaries;
* README/report documentation.

Do NOT commit:

* giant candidate caches;
* image data;
* embedding matrices unless intentionally versioned and reasonably small;
* temporary Parquet;
* model checkpoints unnecessarily;
* secrets.

If an experiment fails and requires a substantial repair, commit the repair separately if it materially improves project infrastructure.

---

# 4. SYSTEM SAFETY / RESOURCE RULES

We previously had disk pressure from DuckDB and large feature caches.

Before heavy work:

* check free disk;
* check RAM;
* cap DuckDB memory/temp usage;
* use batching where needed;
* avoid customer × catalog cross joins;
* do not duplicate dense embeddings across millions of candidate rows;
* use memory mapping / indexed stores / joins by IDs where appropriate;
* periodically clean disposable caches after preserving reports.

If a run crashes:

1. diagnose;
2. repair and retry once;
3. if memory/disk is the issue, use batching/smaller temporary representation;
4. if still broken, record the failure and continue with the next branch.

One failed experiment must NOT terminate the overnight research run.

The LightGBM/PyTorch OpenMP issue was previously solved using process isolation. Preserve or reuse that pattern rather than reintroducing the crash.

---

# 5. FREEZE A REPRODUCIBLE CURRENT BASELINE

Before changing ranking:

Re-run or verify the strongest current baseline under the new development evaluation protocol.

The baseline should approximately be:

```text
Popularity
Repeat
PMI
ALS
Metadata content
Metadata two-tower K=50
       ↓
merge + dedupe
       ↓
LightGBM LambdaRank
       ↓
Top 12
```

Record:

* MAP@12
* Recall@12
* HitRate@12
* candidate recall
* oracle MAP@12
* average candidate count
* latency
* training positives
* number of customers
* number of temporal snapshots
* important LambdaRank hyperparameters

This becomes:

```text
BASELINE_RANKER
```

Every overnight experiment must compare against the same controlled baseline.

---

# 6. STAGE 1 — RANKING FAILURE ANALYSIS

This is mandatory and should guide later branches.

The question is:

> When the correct future-purchased item IS already in the candidate pool, which kinds of positives does LambdaRank systematically bury?

Here:

```text
positive = an item the customer actually buys in the target period
negative = a candidate item that was not purchased in that period
```

Do not confuse "negative" with explicit dislike. These are implicit negatives.

For every retrieved positive, analyze its LambdaRank position.

Build a ranking failure taxonomy.

## A. Popularity bucket

Bucket positive items into something like:

```text
head / very popular
medium
long-tail
```

Prefer quantile-based popularity buckets.

For each bucket report:

* number of positives;
* candidate coverage;
* median rank conditional on retrieval;
* Recall@12 conditional on retrieval;
* MAP/AP contribution;
* source-count distribution.

## B. Source agreement

Bucket positives by:

```text
1 retrieval source
2 sources
3 sources
4+ sources
```

Report median rank and Top-12 rate.

## C. Repeat/novelty relationship

Classify:

* exact repeat item;
* same product type but new article;
* same garment group;
* novel category/product type;
* never previously purchased item.

## D. User history

Bucket by prior transaction count:

```text
0
1–2
3–5
6–10
11–20
20+
```

## E. Item maturity

If feasible:

* high-history established item;
* medium-history;
* new/low-history item.

## F. Retriever provenance

Analyze positives mainly found by:

* repeat
* ALS
* PMI
* content
* two-tower
* popularity
* multi-source agreement.

## G. Positive vs hard-negative feature separability

For difficult positive segments, compare feature distributions against high-ranked negatives.

Example:

```text
long-tail positive
vs
long-tail negative with similar TT score
```

Ask:

> Do our available features actually distinguish them?

Use simple diagnostics:

* histograms/summary stats;
* pairwise margins;
* simple AUC where meaningful;
* SHAP for LambdaRank;
* top false positives / false negatives.

Produce:

```text
artifacts/overnight/ranking_failure_analysis.json
```

and a concise markdown summary.

### Decision logic after Stage 1

If failures are strongly concentrated in a specific segment:

* prioritize experiments targeting that segment.

If positives and negatives are nearly indistinguishable with existing features:

* do NOT assume a new architecture will magically fix the problem.
* emphasize more supervision/data or new information.

If features appear separable but LambdaRank still ranks badly:

* stronger motivation for alternative ranking architecture.

Commit this stage.

---

# 7. STAGE 2 — SCALE RANKING SUPERVISION

This has historically produced the largest gain and therefore gets high priority.

The previous jump:

```text
446 positives → 7,177 positives
```

was extremely successful.

Try to materially increase ranking supervision using more:

* customers;
* temporal snapshots;
* rolling weeks.

Use all feasible historical windows while preserving time safety.

For every snapshot at cutoff T:

```text
history <= T
      ↓
recompute temporally safe retrieval/features
      ↓
future period after T = labels
```

No target-period information may leak into:

* ALS;
* popularity;
* PMI;
* repeat features;
* customer affinities;
* two-tower representations;
* ranker features.

The ranking group must be:

```text
(snapshot_id, customer_id)
```

not just customer_id.

Target:

* substantially more than 7K positives if the dataset permits;
* preferably tens of thousands.

Do NOT blindly produce billions of candidate rows if unnecessary.

Use batching / candidate sampling when required.

Compare:

```text
current supervision baseline
vs
scaled supervision
```

using the same development folds.

Record:

* training positives;
* groups;
* rows;
* training time;
* MAP@12;
* generalization gap.

If scaled data materially improves MAP:

* promote it to the new ranking-data baseline for all later experiments.

Commit.

---

# 8. STAGE 3 — TARGETED SUPERVISION / NEGATIVE EXPERIMENTS

Use Stage 1 diagnosis.

Do not run arbitrary changes without hypotheses.

Possible experiments:

## A. Hard-negative enriched ranking dataset

For each positive, ensure training includes difficult negatives such as:

* high current LambdaRank score but not purchased;
* high ALS score;
* high two-tower score;
* high source agreement;
* semantically/textually plausible if available;
* same category/product type;
* similar popularity bucket.

The purpose is to teach fine distinctions among plausible candidates.

For LightGBM ranking, test controlled candidate downsampling schemes if full 1K+ groups are noisy/expensive.

Compare:

* all candidates;
* top hard negatives + random negatives;
* varying hard-negative proportions.

Do not leak final-fold model predictions into training folds.

Hard negatives must be generated out-of-fold / temporally safely.

## B. Long-tail / rare-positive weighting

ONLY if ranking failure analysis shows a major long-tail problem.

Try small controlled weighting schemes such as:

```text
1 / sqrt(item_popularity)
```

or clipped versions.

Do not aggressively invert popularity.

Select weighting only on earlier folds.

Measure:

* overall MAP@12;
* long-tail Recall@12;
* head-item MAP;
* calibration tradeoff.

Reject if overall MAP deteriorates unless there is a clearly stated multi-objective reason.

## C. Group weighting

Investigate whether very active users / groups with many candidates dominate training.

Test reasonable group/sample weighting if supported correctly.

Do not assume it helps.

---

# 9. STAGE 4 — TARGETED CROSS / FEATURE EXPERIMENTS

LightGBM already learns interactions, but explicit derived features can reveal whether certain relationships are predictive.

Use the ranking failure analysis to select targeted crosses.

Candidate examples:

```text
TT_score × ALS_score
TT_score × source_count
text_score × TT_score
text_score × category_affinity
text_score × inverse_popularity
source_count × popularity_percentile
repeat_score × recency
category_affinity × candidate_category_popularity
ALS_rank - TT_rank
source_rank_agreement
min/mean/max normalized source percentile
```

Only include features for signals that exist in the current frozen baseline or already-computed experimental assets.

Do not restart giant text/image branches just to create dozens of speculative features.

Run an ablation:

```text
baseline features
vs
baseline + targeted interactions
```

Use:

* gain importance;
* SHAP;
* most importantly MAP@12 and temporal-fold consistency.

If explicit crosses do not help LambdaRank, that is evidence that the problem is not merely missing interaction expression.

Commit useful feature changes only.

---

# 10. STAGE 5 — SERIOUS LIGHTGBM RANKING OPTIMIZATION

The goal is not an enormous blind grid search.

Perform a disciplined search on earlier temporal folds.

Inspect current parameters first.

Tune meaningful ranking parameters such as:

* `num_leaves`
* `max_depth`
* `min_data_in_leaf`
* `learning_rate`
* `n_estimators`
* `feature_fraction`
* `bagging_fraction`
* `bagging_freq`
* `lambda_l1`
* `lambda_l2`
* `min_gain_to_split`
* `max_bin`
* `lambdarank_truncation_level`
* `lambdarank_norm`
* early stopping
* `eval_at=[12]` or appropriate list

Pay particular attention to:

> training objective alignment with Top-12 quality.

Test a reasonable truncation level near the region we care about, e.g. around 12–30, rather than defaulting blindly.

If supported in the installed LightGBM version, run a controlled comparison of:

```text
lambdarank
vs
rank_xendcg
```

or other legitimate ranking objectives.

Use temporal cross-fold mean + variance for model selection.

Never use the final holdout for hyperparameter selection.

Preserve a compact search report showing:

* trials;
* parameters;
* fold MAP;
* mean;
* standard deviation;
* runtime.

Promote only robust gains.

---

# 11. STAGE 6 — ALTERNATIVE TREE RANKER

Only after the LightGBM baseline is strong.

Test one serious alternative tree-based ranking family if practical.

Preferred candidate:

```text
CatBoost ranking
```

using an appropriate ranking objective such as YetiRank / YetiRankPairwise if supported.

Reasons:

* handles categorical relationships differently;
* strong regularization/generalization behavior;
* genuinely useful comparison against LambdaMART.

Alternatively use XGBoost ranking only if CatBoost is impractical.

Keep:

* same candidates;
* same folds;
* same features where possible;
* same evaluation metrics.

Do not spend excessive time forcing an alternative tree model to win.

If it clearly loses, record and prune.

---

# 12. STAGE 7 — ONE GENUINELY DIFFERENT RANKING PARADIGM

Do NOT retry DCN as the main experiment.

Instead test a **listwise / set-aware reranker** that solves a different problem.

Architecture concept:

```text
~1000 candidates
      ↓
strong LambdaRank
      ↓
Top N shortlist, e.g. 50–100
      ↓
listwise/set-aware neural reranker
      ↓
Top 12
```

The new model should be able to reason about candidates relative to one another.

Possible implementation:

* compact Transformer/Set Transformer;
* candidate feature embeddings;
* user representation;
* attention across candidate set;
* listwise softmax/ListNet-style objective or another well-justified listwise ranking loss.

Keep it small.

Use LambdaRank to reduce the candidate set so the neural model sees much stronger/harder candidates.

This tests the hypothesis:

> Pointwise/pairwise feature scoring is insufficient; relative candidate context may improve ordering.

### Required comparisons

```text
LambdaRank Top-12
vs
LambdaRank → listwise reranker Top-12
```

Use the same upstream retrieval pool.

Try a small shortlist sweep selected only on earlier folds:

```text
N = 25, 50, 100
```

Do not use final holdout.

Save learning curves.

Run tiny-overfit sanity check first.

If it immediately overfits and loses badly, prune the branch.

---

# 13. STAGE 8 — OPTIONAL GENREC-INSPIRED FINAL RERANKER

Run this ONLY if one of the following is true:

1. Stage 1 shows features contain meaningful distinctions but LambdaRank struggles;
2. listwise reranking shows promising gains;
3. the repository already has a practical foundation-model inference path.

Do NOT spend the whole night installing or downloading giant models if the environment does not already support them well.

The GenRec-inspired architecture should be:

```text
retrieval
   ↓
LambdaRank
   ↓
Top 25–50
   ↓
context-rich foundation-model / transformer reranker
   ↓
Top 12
```

Do NOT run the model across the full catalog or full ~1K pool.

If a true pretrained foundation model is not practical, implement a smaller context-rich transformer reranker and label it honestly as:

```text
GenRec-inspired / context-rich reranker
```

not Netflix GenRec.

Possible context:

### User

* long-term category/garment/price preferences;
* recent purchases;
* existing two-tower user representation;
* purchase frequency;
* relevant semantic summary if cheap.

### Candidate

* metadata;
* retrieval provenance;
* retrieval scores;
* popularity;
* recency;
* two-tower score;
* text description if already cached and safe.

Focus on scoring/ranking, NOT text generation.

No prose decoding is necessary.

If the environment permits a prefilling/scoring-head approach, prefer that.

This branch must earn its compute.

Prune if it is clearly worse than LambdaRank/listwise baseline.

---

# 14. DO NOT REOPEN THESE BRANCHES WITHOUT EVIDENCE

Do not spend overnight cycles blindly retrying:

* larger DCN V2;
* generic image nearest-neighbor retrieval;
* averaged DINO user history;
* unrestricted text candidate expansion;
* history-based TT routing;
* random architecture searches.

These were already tested.

Only revisit if Stage 1 produces a new specific hypothesis.

---

# 15. EXPERIMENT DECISION ENGINE

After every major stage, automatically decide what to do next.

Use rules like:

### If more ranking data produces a strong lift:

Continue scaling until returns clearly diminish, then freeze.

### If long-tail weighting improves only long-tail metrics but hurts MAP:

Reject for primary MAP objective, document tradeoff.

### If targeted crosses improve LambdaRank:

Keep them and carry forward.

### If alternative tree ranker loses clearly:

Prune.

### If listwise reranker gives no meaningful fold-consistent lift:

Do not run an expensive GenRec branch unless diagnostics strongly justify it.

### If listwise reranker wins:

Use its best earlier-fold configuration for final evaluation.

Do not keep multiple complexities for <noise-level gains.

Prefer the simpler model when performance is statistically/fold-wise indistinguishable.

---

# 16. FULL-SCALE RUN

After choosing the best architecture using development folds:

Freeze:

* retrieval sources;
* TT K;
* feature set;
* training snapshots;
* weighting;
* ranker family;
* hyperparameters;
* reranker shortlist size;
* all thresholds.

Then run the largest defensible final evaluation.

If feasible, use the full ~69K customer population or the largest untouched customer cohort.

Process in batches.

Do not tune based on partial final results.

Report:

* MAP@12
* Recall@12
* HitRate@12
* candidate Recall@K
* oracle MAP@12
* average candidate pool size
* latency breakdown
* training time
* peak RAM
* temp disk usage
* model sizes

Also report segment results:

* head/medium/long-tail
* sparse/medium/heavy-history users
* repeat vs novel purchases
* source-count buckets

---

# 17. STATISTICAL / ROBUSTNESS CHECKS

Where practical:

* show metric per temporal fold;
* report mean and std;
* bootstrap customer-level MAP differences for final candidate architectures if feasible;
* avoid declaring a tiny one-off gain a victory.

A model that wins:

```text
+0.0002 MAP on one fold
```

but loses everywhere else is not a robust improvement.

---

# 18. FINAL PROJECT OUTPUTS

By the end of the run, leave the repo with:

## A. Best reproducible architecture

Clearly mark:

```text
BEST_SYSTEM
```

with exact:

* candidate sources;
* budgets;
* feature version;
* training windows;
* ranker;
* reranker if any;
* parameters.

## B. `OVERNIGHT_RESULTS.md`

Create a polished research report containing:

### Executive summary

* best baseline;
* best overnight system;
* absolute and relative lift;
* what actually mattered.

### Experiment timeline

Every meaningful experiment in order.

### Ranking failure analysis

What LambdaRank is good/bad at.

### Data/supervision findings

How training scale affected ranking.

### Feature/interactions findings

### Model comparisons

Include tables such as:

```text
Popularity
LambdaRank baseline
Scaled-data LambdaRank
Best tuned LambdaRank
CatBoost/alternative
Listwise reranker
GenRec-inspired reranker if run
Oracle
```

### Rejected ideas

Explain why each was rejected:

* DCN;
* MLP;
* image retrieval;
* text expansion;
* etc.

### Remaining bottleneck

State whether the remaining limitation appears to be:

* retrieval;
* features/information;
* supervision;
* ranking architecture;
* irreducible purchase uncertainty.

### Production architecture recommendation

### What would change with real H&M production data

Discuss missing:

* impressions;
* clicks;
* carts;
* favorites;
* returns;
* session context;
* inventory;
* availability;
* real-time user state.

## C. README

Update README with the concise stable story, not every debugging detail.

Link to detailed reports.

## D. Artifacts

Preserve small structured reports under something like:

```text
artifacts/overnight/
```

JSON/CSV summaries are encouraged.

Avoid huge caches.

## E. Architecture diagram

Create/update a Mermaid or ASCII architecture showing the final system.

## F. Experiment ledger

Create a table:

```text
experiment
hypothesis
change
development result
final result if applicable
decision
reason
commit hash
```

---

# 19. FINAL GIT STATE

Before finishing:

1. run compilation/lint/tests;
2. verify key pipelines still run;
3. verify no secrets/large temporary files are staged;
4. inspect `git diff`;
5. commit final source/docs/results.

Suggested final commit:

```text
research: finalize overnight recommender experiments and best architecture
```

Do not squash away useful checkpoint history unless repository conventions require it.

At the end, output:

```text
git log --oneline
```

for the overnight commits in the report.

---

# 20. MORNING SUMMARY

At completion, print a concise final message containing:

1. Best architecture.
2. Final MAP@12.
3. Baseline MAP@12.
4. Relative lift.
5. Top 3 things that improved performance.
6. Top 3 things that failed.
7. What the actual bottleneck now appears to be.
8. Whether the project is considered complete.
9. Files to open first:

   * `OVERNIGHT_RESULTS.md`
   * `README.md`
   * key artifact report
10. Git commit hashes/checkpoints.

---

# 21. RESEARCH PRINCIPLES

Follow these throughout:

### Never tune on the final holdout.

### Never claim a result is causal unless the experiment isolates it.

### Never keep complexity merely because it is sophisticated.

### A retrieved item that is not ranked well does not automatically imply architecture failure; diagnose features and supervision.

### Oracle MAP is not an achievable model ceiling because the oracle knows future labels.

### Retrieval recall is valuable only within candidate-quality/latency constraints.

### LightGBM already learns feature interactions; do not retry DCN merely because crosses exist.

### More supervision has historically beaten more model complexity in this project.

### Prefer controlled ablations.

### Preserve failed experiments in documentation.

### If two models are effectively tied, keep the simpler one.

### Do not stop the full overnight run because one branch fails.

The goal is not to make a particular paper/model win.

The goal is to discover the strongest architecture justified by THIS dataset and to understand why.

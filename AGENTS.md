# RecSys Loom agent handoff

Read this file and `README.md` before advising Anshul or changing the repository.
For the current machine-transfer and deployment state, also read `HANDOVER.md`;
it supersedes the dated repository-state notes below.

## Project in one sentence

RecSys Loom rebuilds the evolution of recommendation systems—from popularity and collaborative filtering to multimodal retrieval, learned ranking, and LLM-native recommendation—using real H&M fashion data.

The project is inspired by production systems described by Netflix, Pinterest, YouTube, Airbnb, Etsy, LinkedIn, Uber, Spotify, and other engineering teams. The point is not to imitate their scale. The point is to understand the mechanism behind each idea, implement a locally honest version, and measure what it adds.

## Central thesis

Recommendation systems did not evolve by repeatedly replacing the previous model. They evolved by combining more kinds of evidence inside a staged system while making increasingly explicit tradeoffs around relevance, latency, freshness, constraints, exploration, and user value.

Every additional layer in this repository must address a measured limitation of the previous system and earn its complexity through a controlled experiment.

## Anshul's working boundary

Anshul wants to implement this project himself. Do not put him on autopilot.

Agents may:

- Explain concepts and papers.
- Ask system-design and experimental-design questions.
- Inspect data or code when requested.
- Review his implementation, evaluation logic, and claims.
- Help diagnose a failure without silently replacing his work.
- Suggest a small next milestone and define how to verify it.

Agents must not generate the project ahead of him unless he explicitly asks for implementation. When reviewing, explain the mechanism and let him make the design decision. Avoid dumping a complete solution when a precise question, invariant, test case, or critique will help him learn more.

## Current repository state

As of 27 August 2026:

- The Git repository exists on branch `main` with no commits yet.
- `origin` is `git@github.com-personal:anshulLuhsna/recsys-loom.git`.
- Repository-local Git author is `anshulLuhsna <anshulkalbande@gmail.com>`.
- `README.md` contains the thesis, experiment ladder, evaluation approach, no-autopilot agreement, and Arbityr review workflow.
- `AGENTS.md` is this handoff file.
- There is no implementation yet.
- There is no `.gitignore` yet.
- The raw CSV files are untracked and must not be staged or committed.
- The Arbityr skill is documented but not installed in this repository yet.

Do not claim that a model, evaluator, pipeline, or late Kaggle submission exists until repository evidence confirms it.

## Local data

The repository currently contains:

```text
articles.csv                approximately 35 MB
articles.csv.zip            approximately 5.1 MB
customers.csv               approximately 208 MB
transactions_train.csv      approximately 3.3 GB
```

Observed transaction date range:

```text
2018-09-20 through 2020-09-22
```

The image archive has not been downloaded. `sample_submission.csv` is also absent. Images are intentionally deferred until the transaction-only recommendation loop is trustworthy.

The H&M data provides product metadata, anonymized customers, dated purchases, transaction prices, and sales channels. It does not provide impression logs, clicks, explicit dislikes, natural-language searches, true browsing sessions, inventory, sizes, reservations, warehouses, or authoritative live prices.

Duplicate transaction rows can represent multiple purchases of the same article. Do not delete or deduplicate them without deciding which downstream representation needs quantity and which needs unique relevance labels.

Treat raw data as immutable. Before the first commit, add dataset files and generated artifacts to `.gitignore`. Do not commit or redistribute competition data.

## First prediction task

Given everything known about a customer before a cutoff, rank 12 articles that the customer is likely to purchase during the next seven days.

The first proposed local validation split is:

```text
Training:   2018-09-20 through 2020-09-15
Validation: 2020-09-16 through 2020-09-22
```

No validation-week transaction may influence training features, popularity counts, candidate generation, negative sampling, or model selection.

The official H&M competition evaluates predictions with MAP@12. Local evaluation should also report Recall@12, Hit Rate@12, catalog coverage, popularity concentration, and relevant cold-user/cold-item segments. Later stages may add constraint violations, index freshness, and latency.

An offline score only establishes recovery of later purchases. It does not establish causal lift, satisfaction, or what a customer would have done after seeing a different recommendation slate.

## Immediate milestone

The next milestone is the smallest complete recommendation loop, implemented by Anshul:

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
Reproducible metrics and per-customer predictions
```

The first data audit should establish:

- Row counts and unique customer/article counts.
- Minimum and maximum transaction dates.
- Missing values and schema assumptions.
- Customer and article foreign-key coverage.
- Exact duplicate rows and their interpretation.
- Purchases per customer and per day.
- Product popularity and long-tail shape.

Use DuckDB or Polars for the 3.3 GB transaction file rather than assuming it fits comfortably in pandas memory.

The MAP@12 evaluator must first pass tiny, hand-calculated fixtures. The initial popularity score is an anchor, not a result that needs to look impressive.

Do not begin the frontend, images, collaborative filtering, embeddings, vector databases, learned rankers, or LLM layer before this loop is reproducible.

## Experiment ladder

Add one meaningful capability at a time:

1. Recent and global popularity.
2. Customer repeat-purchase candidates.
3. Item-item co-purchase retrieval.
4. Implicit collaborative filtering.
5. Metadata/content retrieval.
6. Sequential recommendation.
7. Text-image multimodal retrieval.
8. Learned ranking across candidate sources.
9. LLM-assisted intent understanding or supervision.
10. Diversity, availability, freshness, and business-constraint reranking.

For each stage, record the data cutoff, candidate sources, features, model version, metrics, latency where relevant, and per-segment results. Preserve the previous baseline. Do not change the split, metric, and model simultaneously and then attribute the outcome to one cause.

## Production-system lessons to preserve

- A recommender is a staged production system, not one model.
- Retrieval and ranking solve different problems and need separate evaluation.
- Popularity and collaborative signals remain useful even after neural and LLM-based methods are introduced.
- Implicit feedback expresses confidence, not clean preference; an unpurchased item is not automatically a dislike.
- Candidate generation, label construction, negative sampling, recency, and leakage can matter more than architectural novelty.
- Long-term preference, medium-term interest, and immediate intent operate on different timescales.
- LLMs may help interpret intent, create semantic representations, generate supervision, and explain results, but they do not replace catalog grounding, hard filters, deterministic constraints, or efficient serving.
- An unavailable, stale, duplicated, or out-of-catalog recommendation is operationally incorrect even if its relevance score is high.
- Offline accuracy is not equivalent to user satisfaction, causal lift, diversity, or marketplace health.

## Arbityr usage

Arbityr should be used regularly as a repo-grounded decision critic, not as an implementation autopilot.

The project-scoped skill can be installed with:

```bash
gh skill install anshulLuhsna/arbityr-skill arbityr --agent codex --scope project
```

After installation, verify:

```text
.agents/skills/arbityr/SKILL.md
.agents/skills/arbityr/resources/cursor-rule.mdc
```

Run an Arbityr review:

- Before introducing a new candidate source or model family.
- After a baseline or experiment produces results.
- Before changing the evaluation protocol or success metric.
- Before adding expensive infrastructure or synthetic operational data.
- Before publishing a technical claim.

Ground the review in concrete repository evidence. Ask:

1. What measured limitation are we addressing?
2. What is the smallest experiment that can test the idea?
3. What evidence would justify keeping it?
4. What complexity, latency, cost, or reliability tradeoff does it introduce?
5. What result would make us reject or remove it?

Arbityr advises and challenges. Anshul owns the final product and architecture decision.

## Research context

The broader reading list and dataset research live outside this code repository in the PersonalOS vault:

```text
/Users/froncort.ai/Desktop/PersonalOS/PersonalOS/personalOS-vault/Projects/Myntra Shoes/recommender-systems-ai-ml-reading-list.md
/Users/froncort.ai/Desktop/PersonalOS/PersonalOS/personalOS-vault/Projects/Myntra Shoes/myntra-shoes-dataset-options.md
/Users/froncort.ai/Desktop/PersonalOS/PersonalOS/personalOS-vault/Projects/Myntra Shoes/paper notes.md
```

Treat paper and company-blog claims as hypotheses or reported production experience, not automatically replicated truth. Connect each reading to one project decision and the smallest experiment that could test it locally.

## Git and change safety

- Preserve raw data and unrelated user work.
- Do not stage, commit, push, install dependencies, or download large files unless Anshul asks.
- Inspect `git status` before and after changes.
- Never commit raw H&M data, generated model binaries, large prediction artifacts, secrets, or caches.
- Explain material architectural tradeoffs before changing direction.
- Report what changed, what was verified, and what remains unverified.

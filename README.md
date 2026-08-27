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

1. **Popularity:** How far can recency and global demand go?
2. **Repeat purchase:** How much signal exists in a customer's own history?
3. **Item-item retrieval:** Do products purchased together produce better candidates?
4. **Collaborative filtering:** Does shared customer behaviour reveal affinity beyond popularity?
5. **Content retrieval:** Can product metadata improve sparse-user and cold-item recommendations?
6. **Sequential modelling:** Does the order and recency of purchases improve immediate prediction?
7. **Multimodal retrieval:** Do product text and images capture useful fashion similarity?
8. **Learned ranking:** Can the available signals be combined better than fixed rules?
9. **LLM layer:** Does language understanding add measurable value beyond embeddings and conventional rankers?
10. **Constrained reranking:** Can relevance survive diversity, availability, freshness, and latency requirements?

Only one meaningful variable should change between adjacent experiments whenever possible.

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

## Working agreement: no autopilot

Anshul writes the implementation himself. AI tools may:

- Explain concepts and papers.
- Ask design questions.
- Inspect data and code when requested.
- Review experiments, evaluation logic, and claims.
- Help diagnose failures without silently replacing the implementation.

AI tools should not generate the project ahead of Anshul unless he explicitly changes this boundary. The purpose is to develop the underlying engineering judgment, not merely produce a finished repository.

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

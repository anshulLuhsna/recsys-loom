# 22. PRODUCTIZATION — BUILD A DEPLOYABLE DEMO

After the recommendation research program is complete and `BEST_SYSTEM` has been frozen, turn the project into a user-facing application.

The final project must have TWO separate recommendation surfaces:

1. **Personalized Home / For You**
2. **Personalized Product Search**

Do not mock either experience. Both should call real backend retrieval/ranking pipelines.

The purpose is to make the recommender understandable and demoable to another engineer, interviewer, recruiter, or non-technical user.

---

# 23. FRONTEND

First inspect the repository.

If an existing frontend/framework already exists, preserve and extend it.

Otherwise prefer a clean modern stack such as:

```text
Frontend:
Next.js / React + TypeScript

Backend:
FastAPI / existing Python recommendation service
```

Do not rewrite working infrastructure unnecessarily.

The application should look polished enough to deploy publicly.

---

# 24. HOME / PERSONALIZED RECOMMENDATION EXPERIENCE

Create a page such as:

```text
/
or
/recommendations
```

The user should be able to select a demo customer.

Do NOT expose an enormous raw customer ID dropdown with tens of thousands of entries.

Provide:

* a few curated demo customers;
* customer-history bucket labels such as:

  * sparse history
  * medium history
  * heavy history
* optional customer-ID input for advanced testing.

When a customer is selected:

```text
customer
   ↓
BEST_SYSTEM retrieval
   ↓
candidate union
   ↓
LambdaRank / selected final ranker
   ↓
Top 12
```

Display the actual Top-12 recommendations.

Each product card should contain, where available:

* article image;
* product name;
* product type;
* color;
* section;
* garment group;
* price if available;
* useful metadata.

The page should feel like an actual fashion-commerce recommendation surface rather than an ML dashboard.

---

# 25. OPTIONAL “WHY THIS?” DEBUG VIEW

Add an optional expandable/debug mode.

For each recommended product, show simple explanations based on real model features such as:

```text
Popular recently
Similar customers bought this
Matches categories you frequently purchase
Previously purchased similar products
Found by multiple retrieval systems
High two-tower compatibility
```

Do NOT claim these are causal explanations.

Label them as:

```text
Recommendation signals
```

rather than:

```text
Why the AI chose this
```

If helpful, show retrieval provenance:

```text
ALS
PMI
Repeat
Popularity
Content
Two-tower
```

This would make the project much easier to explain in interviews.

---

# 26. SEARCH EXPERIENCE

Create a separate search surface:

```text
/search
```

with a prominent natural-language search box.

Examples:

```text
black oversized hoodie

linen shirt for summer

blue women's dress

minimal black trousers

casual green jacket

white top for office
```

The search architecture should NOT simply search product-name substrings.

Implement a genuine two-stage retrieval/ranking pipeline.

Target conceptual architecture:

```text
USER QUERY
    ↓
QUERY INTENT PARSER
    ↓
structured intent + semantic query
    ↓
HARD FILTERS where appropriate
    ↓
MULTI-SOURCE SEARCH RETRIEVAL
    ↓
hundreds of candidates
    ↓
SEARCH RANKER / FUSION
    ↓
personalized reranking
    ↓
Top results
```

---

# 27. QUERY INTENT PARSER

Build a lightweight query-understanding layer.

Input:

```text
"black oversized hoodie for men"
```

Output conceptually:

```json
{
  "product_type": "hoodie",
  "color": "black",
  "section": "menswear",
  "style_terms": ["oversized"],
  "free_text": "black oversized hoodie for men"
}
```

Possible intent fields:

* product type;
* product group;
* garment group;
* color;
* department/section;
* gender/market segment where represented;
* style terms;
* price intent if dataset supports price;
* free semantic query.

Prefer deterministic / model-light extraction for fields that map directly to known H&M taxonomy.

For example:

```text
"black"
→ color filter/boost

"hoodie"
→ product type / product group
```

An LLM intent parser may be added only if:

* credentials/runtime already make it practical;
* it provides clear value;
* there is a deterministic fallback.

Search must work without requiring an expensive external API.

---

# 28. HARD FILTERS VS SOFT INTENT

Do not treat every extracted term as a hard filter.

Example:

```text
"red dress"
```

`dress` may be a strong filter.

`red` may either be:

* hard filter when confidently mapped;
* strong ranking feature otherwise.

Style terms like:

```text
minimal
streetwear
summer
formal
oversized
```

should usually be semantic/soft relevance features rather than brittle filters.

The intent parser should distinguish:

```text
hard constraints
vs
soft preferences
```

---

# 29. SEARCH RETRIEVAL STAGE

Use several complementary search retrievers.

At minimum test:

## A. Lexical retrieval

Use BM25 or equivalent over useful text:

* product names;
* `detail_desc`;
* product-type names;
* garment groups;
* section names;
* other meaningful searchable metadata.

This catches exact terms.

Example:

```text
"linen shirt"
```

should strongly retrieve products explicitly mentioning linen/shirt.

---

## B. Semantic retrieval

Use the already generated article text embeddings where practical.

Encode the query using the SAME compatible pretrained semantic text encoder.

Then:

```text
query embedding
      ↓
FAISS / ANN
      ↓
semantic candidates
```

Do NOT attempt to compare a raw query embedding against an item representation from a different incompatible embedding space.

Verify query/article embedding compatibility.

---

## C. Structured attribute retrieval

Use parsed intent to retrieve/boost exact matches such as:

```text
color = black
product type = trousers
section = womenswear
```

Avoid customer × catalog brute force.

Use indexes / filtered queries.

---

## D. Personalization candidates

For a logged-in/demo customer, optionally include a small number of candidates from the existing personalized recommender.

However:

$$
\boxed{\text{explicit search intent must dominate personalization}}
$$

If the user searches:

```text
"red dress"
```

do NOT return black trousers merely because the recommender believes the customer likes trousers.

Personalization should refine relevant search results, not override the query.

---

# 30. SEARCH CANDIDATE UNION

Conceptually:

```text
BM25 candidates ─────────┐
Semantic candidates ─────┤
Attribute candidates ────┤
Personalized candidates ─┤
                         ↓
                    MERGE + DEDUPE
                         ↓
                  ~hundreds of items
```

Preserve source information:

```text
bm25_score
bm25_rank

semantic_score
semantic_rank

attribute_match_count

personalization_score

source_count
```

---

# 31. SEARCH RANKING

IMPORTANT:

The H&M Kaggle dataset does NOT contain real search queries, impressions, or query-click/purchase labels.

Therefore, do NOT pretend that the existing recommendation LambdaRank is a properly trained search ranker.

The initial search ranker should optimize query relevance using transparent signals such as:

```text
BM25 score
semantic similarity
exact attribute matches
product-type match
color match
section match
popularity
personalization compatibility
```

A sensible initial architecture is:

```text
query relevance
        +
small personalization boost
        +
quality/popularity priors
```

Explicit query relevance must have the highest priority.

---

# 32. HYBRID SEARCH FUSION BASELINE

Implement a robust hybrid baseline first.

Possible approach:

```text
Reciprocal Rank Fusion
```

or normalized weighted scoring between:

```text
BM25
semantic retrieval
structured matches
```

Example conceptual score:

```text
search_score =
    query_relevance
    + small_personalization_component
```

Do not hard-code arbitrary weights without testing on development queries.

---

# 33. SEARCH EVALUATION WITHOUT REAL SEARCH LOGS

Because no real query logs exist, build an honest offline search test suite.

Do NOT claim this is equivalent to production search evaluation.

Generate/curate a set of representative search queries from article metadata/descriptions.

Examples:

```text
black trousers
blue dress
men's hoodie
oversized jacket
white cotton shirt
red cardigan
green casual top
black sportswear
```

Create evaluation labels using only defensible metadata/semantic rules.

For exact structured queries:

```text
"black trousers"
```

articles matching:

```text
color=black
AND product_type≈trousers
```

can be considered relevant for the synthetic benchmark.

For semantic/style queries, use weaker evaluation and manual spot checking.

Report metrics such as:

```text
Recall@50
Precision@10
NDCG@10
MRR
```

but label them:

```text
synthetic search benchmark
```

NOT:

```text
real user search quality
```

---

# 34. OPTIONAL WEAKLY SUPERVISED SEARCH RANKER

Only if the hybrid baseline is working and enough time remains:

Create a synthetic/weakly supervised learning-to-rank experiment.

Possible training data:

```text
synthetic query
+
article candidate
+
relevance label derived from metadata/text
```

Train a small:

```text
LightGBM LambdaRank
```

search ranker using query-item features.

Do NOT reuse the personalized home-feed LambdaRank blindly.

Search ranker groups should be:

```text
query_id
```

rather than customer_id.

Possible features:

```text
BM25 score
semantic similarity
exact product-type match
exact color match
section match
garment-group match
query token overlap
popularity
personalization score
```

Evaluate only on held-out synthetic queries.

Keep it only if it robustly beats hybrid fusion.

Document clearly that this ranker is trained with synthetic relevance because real search logs are unavailable.

---

# 35. PERSONALIZED SEARCH RERANKING

For a selected customer:

```text
query
  ↓
search retrieval
  ↓
query-relevant candidate pool
  ↓
query relevance ranking
  ↓
personalization refinement
  ↓
results
```

Useful personalization features may include:

```text
ALS compatibility
metadata TT score
category affinity
repeat behavior
customer price/category preference
```

But cap personalization so it cannot destroy explicit query relevance.

A good mental model:

```text
SEARCH INTENT = primary
USER PREFERENCE = secondary
```

Test both:

```text
anonymous search
personalized search
```

The frontend should let the user toggle/select a customer so the difference is visible.

---

# 36. SEARCH API

Expose a backend endpoint such as:

```text
GET /api/search
```

or:

```text
POST /api/search
```

Inputs:

```json
{
  "query": "black oversized hoodie",
  "customer_id": "...",
  "limit": 20
}
```

Return:

```json
{
  "query": "...",
  "parsed_intent": {},
  "results": [],
  "latency_ms": {
    "intent": 0,
    "retrieval": 0,
    "ranking": 0,
    "total": 0
  }
}
```

Each result should contain:

* article ID;
* product metadata;
* image path/URL;
* final search score;
* optional debug retrieval signals.

---

# 37. RECOMMENDATION API

Expose something like:

```text
GET /api/recommendations/{customer_id}
```

Return Top 12 from `BEST_SYSTEM`.

Include:

```text
model version
candidate count
latency
recommendations
```

Do not retrain models during requests.

Load models/indices once at application startup.

Cache common data.

---

# 38. PERFORMANCE

The deployed application should NOT:

* load Parquet files from scratch on every request;
* retrain ALS;
* retrain LambdaRank;
* rebuild FAISS;
* recompute all customer features.

Prepare deployable artifacts offline.

At service startup:

```text
load model
load item metadata
load retrieval indices
load required user features/embeddings
```

Then serve requests efficiently.

Measure approximate endpoint latency.

---

# 39. FRONTEND SEARCH UI

The `/search` experience should contain:

* large search bar;
* optional customer selector;
* search results grid;
* image cards;
* useful filters;
* result count;
* loading state;
* empty state;
* error handling.

Optionally show:

```text
Parsed intent:
Black
Hoodie
Menswear
Oversized
```

as small chips.

This makes the query-understanding pipeline visible.

---

# 40. HOMEPAGE UI

Create a polished landing page.

Suggested sections:

```text
For You
```

Top-12 personalized recommendations.

```text
Search H&M
```

Natural-language search.

Optionally:

```text
Trending
```

to demonstrate the popularity baseline.

The page should visually communicate the difference between:

```text
personalized recommendation
vs
explicit search
```

---

# 41. PRODUCT DETAIL / MODAL

If practical, clicking a product should open:

* larger image;
* product description;
* metadata;
* recommendation/search signals.

No need to recreate the full H&M website.

Keep scope focused on demonstrating the recommender.

---

# 42. DEMO MODE

Create a small set of curated demo personas/customers.

For example:

```text
Sparse-history customer
Streetwear-heavy customer
Womenswear-heavy customer
Frequent shopper
Long-tail-oriented customer
```

These must correspond to real anonymized dataset customers.

Do not invent transaction histories.

Give them display names like:

```text
Demo Customer A
Demo Customer B
```

and describe behavior using derived aggregate statistics.

This makes the app easy to demo without exposing giant customer IDs.

---

# 43. DEPLOYMENT

Prepare the project so it can be deployed.

At minimum provide:

* frontend build instructions;
* backend run instructions;
* environment-variable template;
* Dockerfiles where appropriate;
* `.gitignore`;
* startup commands;
* model/artifact download/build instructions;
* deployment README.

Prefer a structure like:

```text
/apps/web
/services/recommender
/artifacts/deploy
```

only if refactoring is actually useful.

Do not reorganize the whole repo gratuitously.

If a single Docker image or Docker Compose setup is cleaner, use that.

---

# 44. DEPLOYMENT ARTIFACT SIZE

Be mindful that the raw H&M image set is huge.

Do NOT assume a deployment platform can ship tens of gigabytes of raw data.

Create a deployable subset/strategy.

Possible approaches:

* serve images from an external/static source if existing paths are legally/technically usable;
* include only images required for demo inventory;
* create optimized thumbnails;
* maintain metadata/index artifacts separately.

Document exactly what is required to reproduce the full local version versus the public demo.

Do not commit gigantic image folders to Git.

---

# 45. SEARCH LIMITATION DOCUMENTATION

README must explicitly say:

> Personalized home recommendations are trained/evaluated using historical H&M purchase data.

and:

> Search is a hybrid retrieval/ranking demonstration built from product metadata and semantic representations. Because the H&M dataset contains no real search-query/click logs, search relevance is evaluated using synthetic/structured benchmarks rather than production search behavior.

This distinction is important.

---

# 46. PRODUCT ARCHITECTURE DIAGRAM

Update the final architecture diagram to show BOTH paths.

Something like:

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
      LambdaRank                      merge/dedupe
          │                                │
          ▼                                ▼
       Top 12                     query relevance rank
                                           │
                                           ▼
                                  personalization rerank
                                           │
                                           ▼
                                      search results
```

---

# 47. FRONTEND/PRODUCT GIT CHECKPOINTS

Continue periodic commits.

Suggested productization commits:

```text
feat: add deployable recommendation API

feat: add hybrid product search pipeline

feat: add personalized search reranking

feat: build recommendation and search frontend

docs: add deployment and search evaluation guide

checkpoint: complete deployable recommender demo
```

Do not mix all frontend/backend/search work into one enormous commit.

---

# 48. FINAL MORNING DEFINITION OF DONE

The project is only considered fully complete if the morning result includes:

### ML / research

* best recommendation architecture;
* final ranking metrics;
* experiment ledger;
* ranking failure analysis;
* documented rejected approaches.

### Recommendation product

* real Top-12 recommendation endpoint;
* working personalized recommendation frontend.

### Search product

* natural-language query input;
* query intent parsing;
* lexical retrieval;
* semantic retrieval;
* structured filtering;
* two-stage search ranking/fusion;
* optional personalization;
* synthetic search evaluation.

### Engineering

* reproducible models/indices;
* deployment artifacts;
* Docker/setup instructions;
* tests;
* lint;
* reasonable latency;
* no giant temp files committed.

### Documentation

* `README.md`
* `OVERNIGHT_RESULTS.md`
* search architecture/evaluation section;
* deployment instructions;
* final architecture diagram.

### Git

* meaningful periodic commits;
* clean final `git status`;
* final commit hash list.

The frontend is not an optional bonus.

It is part of the overnight project completion criteria.

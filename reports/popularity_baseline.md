# Global popularity baseline

## Experiment contract

- Popularity score: count of purchase rows per article.
- Training data: transactions dated on or before 2020-09-15.
- Validation data: transactions dated 2020-09-16 through 2020-09-22.
- Predictions: the same 12 training-popular articles for every validation customer.
- Tie-break: `article_id` ascending.
- Relevance labels: unique `(customer_id, article_id)` pairs in validation.
- Catalog denominator: all 105,542 articles in `articles.csv`.
- Evaluated customers: all 68,984 customers with at least one validation purchase.

No validation row contributes to the popularity counts.

## Top 12 from training

| Rank | Article | Training purchases |
| ---: | --- | ---: |
| 1 | 0706016001 | 49,958 |
| 2 | 0706016002 | 34,802 |
| 3 | 0372860001 | 31,482 |
| 4 | 0610776002 | 30,003 |
| 5 | 0759871002 | 26,309 |
| 6 | 0464297007 | 24,989 |
| 7 | 0372860002 | 24,222 |
| 8 | 0610776001 | 22,335 |
| 9 | 0399223001 | 22,204 |
| 10 | 0720125001 | 21,005 |
| 11 | 0706016003 | 20,957 |
| 12 | 0156231001 | 20,884 |

## Results

Metrics are macro-averaged across customers unless marked micro. Decimal values are retained beside percentages to prevent visual rounding from hiding the exact result.

| Segment | Customers | Customers with hit | Matched relevance pairs | MAP@12 | Recall@12 | Micro Recall@12 | Hit Rate@12 | Catalog coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All | 68,984 | 1,457 | 1,635 | 0.00289859 | 0.00774900 (0.7749%) | 0.00764991 (0.7650%) | 0.02112084 (2.1121%) | 0.00011370 (0.0114%) |
| Warm customers | 63,412 | 1,330 | 1,490 | 0.00288274 | 0.00777616 (0.7776%) | 0.00754381 (0.7544%) | 0.02097395 (2.0974%) | 0.00011370 (0.0114%) |
| Cold customers | 5,572 | 127 | 145 | 0.00307900 | 0.00743992 (0.7440%) | 0.00894234 (0.8942%) | 0.02279253 (2.2793%) | 0.00011370 (0.0114%) |

## Interpretation

The baseline finds at least one later-purchased article for 1,457 of 68,984 customers. It recovers 1,635 of 213,728 unique validation relevance pairs. This is deliberately a weak, non-personalized anchor: every customer receives the same list, and only 12 of 105,542 catalog articles are ever recommended.

The slightly higher cold-customer MAP and Hit Rate do not show that popularity handles cold users better. The prediction list is identical for both segments; the difference reflects the articles and target-set sizes purchased by those groups during this validation week.

This result establishes offline recovery of later purchases only. It does not establish causal lift, satisfaction, availability, or operational correctness.

## Verification

- Eight hand-calculated unit and tiny end-to-end tests pass.
- Fixtures cover perfect ranking, no hit, rank sensitivity, duplicate predictions, the rank-12 cutoff, more than 12 relevant items, aggregate metrics, and leakage-safe training counts.
- An independent DuckDB query reproduced exactly 1,635 matched relevance pairs and 1,457 customers with a hit.
- Machine-readable metrics and per-customer predictions are stored under `artifacts/popularity_baseline/` and ignored by Git.

## Next question

The next experiment should change only the popularity time horizon: compare this all-history anchor with one predeclared recent window while keeping the split, labels, metric implementation, tie-breaking, and prediction count fixed.

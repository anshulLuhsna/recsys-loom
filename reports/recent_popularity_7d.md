# Seven-day recent-popularity experiment

## Question

Does counting only the final seven training days recover more purchases in the following validation week than counting the complete two-year history?

This experiment changes only the popularity window. The evaluator, validation customers, hidden labels, top-12 limit, article-level counting, duplicate interpretation, and tie-breaking rule remain fixed.

## Experiment contract

- Popularity window: 2020-09-09 through 2020-09-15, inclusive.
- Validation window: 2020-09-16 through 2020-09-22, inclusive.
- Popularity score: purchase-row count per article within the seven-day training window.
- Tie-break: `article_id` ascending.
- Relevance: unique `(customer_id, article_id)` pairs in validation.
- Evaluated population: the same 68,984 validation customers as the all-history baseline.

## Recent top 12

| Rank | Article | Seven-day purchases |
| ---: | --- | ---: |
| 1 | 0909370001 | 1,283 |
| 2 | 0865799006 | 768 |
| 3 | 0918522001 | 729 |
| 4 | 0924243001 | 704 |
| 5 | 0448509014 | 609 |
| 6 | 0751471001 | 607 |
| 7 | 0809238001 | 563 |
| 8 | 0918292001 | 546 |
| 9 | 0762846027 | 539 |
| 10 | 0809238005 | 503 |
| 11 | 0673677002 | 463 |
| 12 | 0923758001 | 457 |

None of these articles appears in the all-history top 12. That complete turnover is evidence that the full-history list was dominated by durable historical demand rather than the catalog's immediate purchase trend.

## Results

| Metric | All history | Recent seven days | Absolute change | Relative change |
| --- | ---: | ---: | ---: | ---: |
| MAP@12 | 0.00289859 | 0.00874768 | +0.00584909 | +201.79% |
| Macro Recall@12 | 0.00774900 | 0.02550429 | +0.01775529 | +229.13% |
| Micro Recall@12 | 0.00764991 | 0.02287955 | +0.01522964 | +199.08% |
| Hit Rate@12 | 0.02112084 | 0.06430477 | +0.04318393 | +204.46% |
| Catalog coverage | 0.00011370 | 0.00011370 | 0 | 0% |

The recent list matched 4,890 unique validation relevance pairs for 4,436 customers. Compared with all-history popularity, that is 3,255 additional matched pairs and 2,979 additional customers with at least one hit.

Segment results:

| Segment | Customers | MAP@12 | Recall@12 | Hit Rate@12 |
| --- | ---: | ---: | ---: | ---: |
| Warm customers | 63,412 | 0.00882410 | 0.02553561 | 0.06516117 |
| Cold customers | 5,572 | 0.00787797 | 0.02514783 | 0.05455851 |

## Decision review

1. **Measured limitation:** The all-history model gives a September 2018 purchase the same weight as a September 2020 purchase. Its top 12 matched only 1,635 validation pairs.
2. **Smallest experiment:** Replace the all-history count with one predeclared seven-day count while holding every other evaluation choice fixed.
3. **Evidence for keeping it:** On this frozen week, MAP@12 is 3.018 times the all-history score and Hit Rate@12 is 3.045 times the all-history score.
4. **Tradeoff introduced:** A seven-day count is more responsive but more volatile. Promotions, payday effects, stock changes, or a single unusual week can dominate it.
5. **Rejection condition:** Reject a general claim that seven days is better if the improvement does not survive additional frozen chronological validation weeks.

## Conclusion boundary

For the 2020-09-16 through 2020-09-22 holdout, recent demand is substantially more predictive than unweighted all-history demand. This supports preserving the seven-day result as the stronger single-week popularity baseline.

It does not yet establish that seven days is the best lookback or that the improvement is stable over time. No 30-day window, decay function, or additional week has been tested. Choosing among those is a future experiment, not a conclusion from this run.

## Verification

- A new tiny-data test proves that `training_start` excludes older high-count purchases.
- All nine evaluator and baseline tests pass.
- An independent DuckDB query reproduced exactly 4,890 matched pairs and 4,436 customers with a hit.
- Machine-readable metrics and predictions are stored under `artifacts/recent_popularity_7d/` and ignored by Git.

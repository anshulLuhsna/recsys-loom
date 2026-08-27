# Eight-week popularity backtest

## Question

Does the seven-day popularity improvement observed on 2020-09-16 through 2020-09-22 survive additional chronological validation weeks?

## Contract

- Eight non-overlapping Wednesday-through-Tuesday validation weeks.
- Oldest validation week: 2020-07-29 through 2020-08-04.
- Newest validation week: 2020-09-16 through 2020-09-22.
- Each fold trains only through the day before its validation week.
- All-history model: counts every purchase through that fold's training cutoff.
- Recent model: counts only the final seven training days before that fold.
- Both models recommend 12 articles to the same validation customers.
- Labels are unique `(customer_id, article_id)` pairs within each validation week.
- Evaluation code, tie-breaking, duplicate handling, and catalog denominator remain fixed.

## Per-week results

| Validation week | Customers | All-history MAP@12 | Recent MAP@12 | MAP ratio | All-history Recall@12 | Recent Recall@12 | All-history Hit Rate | Recent Hit Rate | Top-12 overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2020-07-29–2020-08-04 | 82,802 | 0.00372181 | 0.00376817 | 1.012× | 0.01134838 | 0.01417190 | 0.03288568 | 0.04353760 | 5 |
| 2020-08-05–2020-08-11 | 74,833 | 0.00212500 | 0.00248403 | 1.169× | **0.00769991** | 0.00677120 | **0.02417383** | 0.02184865 | 4 |
| 2020-08-12–2020-08-18 | 71,094 | 0.00295010 | 0.00317412 | 1.076× | 0.00908472 | 0.01041380 | **0.02745661** | 0.02730188 | 0 |
| 2020-08-19–2020-08-25 | 72,035 | 0.00372497 | 0.00786726 | 2.112× | 0.00938505 | 0.02063695 | 0.02722288 | 0.05622267 | 1 |
| 2020-08-26–2020-09-01 | 80,253 | 0.00402862 | 0.00682845 | 1.695× | 0.00880784 | 0.02326702 | 0.02490873 | 0.05805390 | 1 |
| 2020-09-02–2020-09-08 | 75,822 | 0.00381272 | 0.00673213 | 1.766× | 0.00866969 | 0.02553012 | 0.02416185 | 0.06432170 | 1 |
| 2020-09-09–2020-09-15 | 72,019 | 0.00344401 | 0.00660410 | 1.918× | 0.00897065 | 0.02090514 | 0.02477124 | 0.05651286 | 1 |
| 2020-09-16–2020-09-22 | 68,984 | 0.00289859 | 0.00874768 | 3.018× | 0.00774900 | 0.02550429 | 0.02112084 | 0.06430477 | 0 |

Bold values mark the two cases where all-history popularity beats seven-day popularity on a secondary metric.

## Pooled results

The eight folds contain 597,842 customer-week evaluations and 1,950,081 unique relevance pairs.

| Metric | All history | Recent seven days | Relative change |
| --- | ---: | ---: | ---: |
| Customer-weighted MAP@12 | 0.00335481 | 0.00573362 | +70.91% |
| Customer-weighted macro Recall@12 | 0.00900342 | 0.01835779 | +103.90% |
| Micro Recall@12 | 0.00863554 | 0.01615420 | +87.07% |
| Hit Rate@12 | 0.02595502 | 0.04896444 | +88.65% |
| Customers with a hit | 15,517 | 29,273 | +13,756 |
| Matched relevance pairs | 16,840 | 31,502 | +14,662 |

Weekly win counts:

- MAP@12: recent wins 8 of 8.
- Macro Recall@12: recent wins 7 of 8.
- Micro Recall@12: recent wins 7 of 8.
- Hit Rate@12: recent wins 6 of 8.

## Interpretation

The original holdout was not an isolated MAP improvement. Seven-day popularity beats all-history popularity on MAP@12 in every tested week, ranging from a 1.012× improvement in the oldest fold to 3.018× in the newest fold. Because MAP@12 is the declared primary metric, the evidence supports keeping seven-day popularity as the stronger current baseline.

The result is not universal dominance. During 2020-08-05 through 2020-08-11, seven-day popularity ranks its fewer correct predictions better, producing higher MAP while losing Recall and Hit Rate. During the following week, its Hit Rate is marginally lower despite higher MAP and Recall. A short window can therefore improve ordering while occasionally narrowing the number of customers reached.

The top-12 overlap varies from zero to five articles, showing that recent popularity is substantially more volatile than all-history popularity. That volatility is the cost of freshness and motivates testing a less reactive recency treatment rather than assuming seven days is optimal.

## Decision review

1. **Measured limitation addressed:** All-history counts are stale relative to the next-week prediction task.
2. **Smallest test:** Eight fixed weekly folds comparing only the count window.
3. **Evidence for keeping recent popularity:** MAP@12 wins in all eight folds and pooled MAP improves by 70.91%.
4. **Cost introduced:** Complete or near-complete top-list turnover and two weeks with a secondary-metric loss.
5. **Evidence that would reverse the decision:** Material MAP losses on additional seasonal periods, or a less volatile recency method matching MAP while improving reach.

## Conclusion boundary

Across these eight consecutive late-summer 2020 folds, seven-day popularity is the stronger MAP@12 baseline. The folds are consecutive rather than seasonally diverse, so this does not establish year-round stability or prove that seven days is the best possible lookback.

The next controlled question is whether a 30-day window or a simple recency decay preserves most of the MAP gain while reducing week-to-week list turnover and secondary-metric losses.

## Verification

- Ten tests pass, including exact chronological-window boundaries and exclusion of older counts.
- Both models use identical customers and relevance counts within every fold.
- Pooled statistics were independently recomputed from the stored per-fold results.
- The full run completed in 158.7 seconds using two DuckDB threads.
- Results and each fold's machine-readable metrics are stored under `artifacts/popularity_backtest_8w/` and ignored by Git.

# H&M data audit findings

These conclusions interpret the exact tables in `reports/data_audit.md`. They describe the local files and the proposed temporal evaluation task; they do not establish recommendation lift or customer satisfaction.

## Executive summary

The three files are structurally usable for the first transaction-only recommendation loop. Transaction dates and identifiers parse cleanly, catalog keys are unique, and every customer and article referenced by a transaction exists in its corresponding catalog. The main modeling risks are not corrupt keys. They are sparse histories, missing customer attributes, repeated purchase rows, popularity skew, and cold-start cases in the validation week.

## Findings and consequences

1. **The relational core is clean.** There are 31,788,324 transactions from 1,362,281 customers involving 104,547 articles. Transaction identifiers have consistent lengths, and no transaction references an unknown customer or article. This makes a transaction-only baseline feasible without repairing keys first.

2. **Some catalog entities have no purchase history.** The customer catalog contains 9,699 customers absent from the complete transaction file, and the article catalog contains 995 articles absent from it. These rows should remain available for catalog and cold-start analysis; they should not silently enter popularity denominators as previously observed items.

3. **Customer metadata cannot be treated as complete.** `FN` is missing for 65.24% of customers and `Active` for 66.15%. Age is missing for 1.16%, club status for 0.44%, and fashion-news frequency for 1.17%. Missing `FN` or `Active` must remain a separate state unless the data documentation establishes that null means false.

4. **Customer categories need normalization before modeling.** Fashion-news frequency contains both `NONE` and `None`; the latter appears only twice. Normalize spelling and case in derived features, while keeping the raw file immutable and recording the mapping.

5. **Article metadata is nearly complete.** Only 416 articles lack `detail_desc`; other inspected article fields are present. The 105,542 article variants map to 47,224 product codes, so later content experiments must decide whether similarity operates at product-code or article-variant level.

6. **Exact duplicate rows are common and should not be deleted blindly.** There are 2,543,908 duplicate groups containing 5,518,813 rows. Removing all repeats would discard 2,974,905 rows, or 9.36% of transactions. Most duplicate groups occur twice, but the largest occurs 569 times. Preserve row frequency for quantity-sensitive features; construct MAP@12 relevance from unique `(customer_id, article_id)` pairs within the evaluation window.

7. **The typical customer history is sparse.** Among customers with transactions, the median history is nine purchases across three active days. The mean is 23.33 purchases, the 99th percentile is 187, and the maximum is 1,895. Report results by history-size segment because an aggregate score will mix fundamentally different recommendation conditions.

8. **Popularity is long-tailed but not completely dominated by a few products.** The median purchased article has 65 transactions, while the most-purchased article has 50,287. The top 12 articles account for 1.04% of transactions, the top 100 for 4.60%, and the top 1,000 for 18.16%. A popularity baseline is a useful anchor, but its coverage and concentration should be measured explicitly.

9. **Daily volume varies materially.** Across 734 observed days, median volume is 39,515.5 transactions and the maximum is 198,622. Peaks include late-November dates and other isolated spikes. A recent-popularity baseline should therefore state its lookback window and whether it uses raw counts, decay, or day normalization.

10. **The proposed split is cleanly representable.** Training contains 31,548,013 transactions through 2020-09-15. Validation contains 240,311 transactions from 68,984 customers during 2020-09-16 through 2020-09-22. The validation week is only 0.756% of all transactions, so no validation row may contribute to popularity counts or feature construction.

11. **Validation includes genuine cold-start cases.** Of the validation customers, 5,572 (8.08%) have no training purchase. Of the 17,986 validation articles, 667 (3.71%) do not appear in training. Report warm-user and cold-user results separately. A transaction-only popularity method has no behavioral evidence for a cold article, so cold-item performance should not be presented as comparable to warm-item recovery.

12. **Validation labels must be deduplicated.** The validation file contains 240,311 transaction rows but only 213,728 unique customer-article relevance pairs. The difference is 26,583 repeated rows, or 11.06% of validation transactions. Each article should count once as relevant to a customer when calculating MAP@12, Recall@12, and Hit Rate@12.

13. **MAP@12 will evaluate customers with different target-set sizes.** The median validation customer purchased two distinct articles, the 90th percentile purchased six, the 99th percentile purchased fourteen, and the maximum is fifty. Retain the complete unique relevant set in the denominator even though predictions are capped at twelve.

## Decisions for the first baseline

- Parse `customer_id` and `article_id` as strings.
- Treat the CSV files as immutable inputs.
- Train only on transactions dated on or before 2020-09-15.
- Build validation relevance as unique `(customer_id, article_id)` pairs from 2020-09-16 through 2020-09-22.
- Preserve duplicate frequency in the raw interaction representation; do not let duplicates multiply binary relevance labels.
- Define a deterministic strategy for cold customers, most likely the same training-only popularity list used as the global fallback.
- Report warm versus cold customers and warm versus cold articles alongside the aggregate metrics.
- Record the popularity lookback and tie-breaking rule so predictions are reproducible.

## Still unverified

- Whether identical transaction rows definitively encode multiple units rather than source duplication.
- Whether missing `FN` and `Active` values have a documented semantic meaning.
- Whether the normalized `price` field should contribute to early baselines; it is not an authoritative live price.
- Whether the single proposed validation week is representative. Multiple frozen chronological weeks should be added only after the first loop works end to end.

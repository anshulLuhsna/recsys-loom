# Experiment ledger

Development metric: mean MAP@12 on the 2020-08-31 and 2020-09-07 2K folds.
Final metric: reserved unused-customer holdout on 2020-09-16..22. Do not inspect it during selection.

| experiment | hypothesis | change | development result | final result | decision | reason | commit |
|---|---|---|---|---|---|---|---|
| evaluation integrity | previously viewed Sep 16–22 metrics are not a pristine holdout | ledger + unused-customer reservation | no unused calendar week; 8,000 unused buyers reserved | not inspected | keep customer-holdout protocol | dataset ends 2020-09-22 | 0aa49c6 |
| BASELINE_RANKER | current six-source LambdaRank is the controlled starting point | freeze TT K=50 + existing LightGBM params | mean MAP@12 0.02707 (0.02638 / 0.02776) | not for selection | keep | matches prior TT-K=50 earlier-fold mean | cbbfe17 |
| ranking failure analysis | complementary retrievers find positives the ranker cannot promote | taxonomy on OOF LambdaRank ranks | median retrieved-positive rank 101; 15.6% reach Top 12; repeats 61% vs ALS-only 1.8% | not for selection | keep diagnosis; do not retry DCN | features barely separate buried long-tail positives | pending |

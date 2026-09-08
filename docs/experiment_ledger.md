# Experiment ledger

Development metric: mean MAP@12 on the 2020-08-31 and 2020-09-07 2K folds.
Final metric: reserved unused-customer holdout on 2020-09-16..22. Do not inspect it during selection.

| experiment | hypothesis | change | development result | final result | decision | reason | commit |
|---|---|---|---|---|---|---|---|
| evaluation integrity | previously viewed Sep 16–22 metrics are not a pristine holdout | ledger + unused-customer reservation | no unused calendar week remains | pending | keep customer-holdout protocol | dataset ends 2020-09-22 | pending |
| BASELINE_RANKER | current six-source LambdaRank is the controlled starting point | freeze TT K=50 + existing LightGBM params | pending | not for selection | pending | pending | pending |

# Experiment ledger

Development metric: mean MAP@12 on the 2020-08-31 and 2020-09-07 2K folds.
Final metric: reserved unused-customer holdout on 2020-09-16..22. Do not inspect it during selection.

| experiment | hypothesis | change | development result | final result | decision | reason | commit |
|---|---|---|---|---|---|---|---|
| evaluation integrity | previously viewed Sep 16–22 metrics are not a pristine holdout | ledger + unused-customer reservation | no unused calendar week; 8,000 unused buyers reserved | not inspected | keep customer-holdout protocol | dataset ends 2020-09-22 | 0aa49c6 |
| BASELINE_RANKER | current six-source LambdaRank is the controlled starting point | freeze TT K=50 + existing LightGBM params | mean MAP@12 0.02707 (0.02638 / 0.02776) | not for selection | keep | matches prior TT-K=50 earlier-fold mean | cbbfe17 |
| ranking failure analysis | complementary retrievers find positives the ranker cannot promote | taxonomy on OOF LambdaRank ranks | median retrieved-positive rank 101; 15.6% reach Top 12; repeats 61% vs ALS-only 1.8% | not for selection | keep diagnosis; do not retry DCN | features barely separate buried long-tail positives | c269086 |
| LightGBM truncation | aligning the objective with Top-12 helps | lambdarank_truncation_level 12 / 20 / 30 | 0.02313 / 0.02538 / 0.02707 vs baseline 0.02707 | not inspected | reject | worse or tied; keep default truncation | bc53c24 |
| LightGBM slower schedule | lower lr + more trees beats the 300/0.05 schedule | lr=0.03, 500 trees | 0.02799 vs baseline 0.02707 | not inspected | **keep** | uniquely best; +0.00092 over baseline | pending |
| LightGBM min_child | less/more regularization helps | min_child 20 / 100 | 0.02600 / 0.02639 | not inspected | reject | both below baseline | pending |
| LightGBM XENDCG | XENDCG beats LambdaRank on MAP@12 | objective=rank_xendcg | 0.02346 | not inspected | reject | worse than LambdaRank | pending |
| LightGBM L2 | lambda_l2=1.0 regularizes usefully | lambda_l2=1.0 on 300-tree control | 0.02774 | not inspected | reject | below selected 0.02799 by 0.00025 | pending |
| hard-negative downsampling | hard+random negatives teach finer distinctions than full 1K+ groups | hard50/rand50, hard100/rand50, hard50/rand150 | 0.01878 / 0.01815 / 0.02204 vs 0.02799 | not inspected | reject | full groups win by a wide margin | pending |
| group weighting | large groups / active users dominate training | 1/sqrt(group size) and 1/sqrt(history) | queued in finalize | not inspected | pending | Stage 3C from overnight prompt | pending |
| hybrid search baseline | metadata + BM25 can answer structured catalog queries | intent + BM25 + structured (+ optional MiniLM) | synthetic structured P@10=1.0 (tautological); style P@10 0.667 / NDCG 0.775 | n/a | keep as search baseline | no query logs; label as synthetic | c67d77d |

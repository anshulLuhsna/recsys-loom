# Evaluation ledger

This is the overnight evaluation-integrity record. It is not a pristine unused
calendar holdout. The transaction file ends on 2020-09-22, and every previously
reported “final week” already used 2020-09-16 through 2020-09-22.

Machine-readable copy: `artifacts/overnight/evaluation_ledger.json`.

## Protocol

| Role | Window | Customers | Status |
|---|---|---|---|
| Training / features | history ≤ cutoff T | 2K per snapshot | Safe if T is before the target week |
| Development selection | 2020-08-31 → Sep 1–7 | 2K, `ORDER BY customer_id` | Used previously for two-tower K; still the development metric |
| Development selection | 2020-09-07 → Sep 8–14 | 2K, `ORDER BY customer_id` | Same as above |
| Contaminated reference | 2020-09-15 → Sep 16–22 | 100 / 500 / 1K / 2K / 5K | Inspected many times; **not for selection** |
| Official final | 2020-09-15 → Sep 16–22 | 8,000 reserved unused buyers | Customer holdout; inspect once after freeze |

There is **no unused temporal window**. Final claims must be labeled
**customer-holdout final evaluation**, not unseen-week test.

## Weekly buyer populations

| Target week | Customers | Transactions | Articles |
|---|---:|---:|---:|
| 2020-08-18 – 24 | 70,416 | 249,717 | 19,932 |
| 2020-08-25 – 31 | 80,948 | 286,961 | 19,698 |
| 2020-09-01 – 07 | 76,556 | 266,302 | 19,573 |
| 2020-09-08 – 14 | 74,575 | 265,603 | 18,674 |
| 2020-09-16 – 22 | 68,984 | 240,311 | 17,986 |

## How customers were previously sampled

- Ranking caches: `ORDER BY customer_id LIMIT N`.
- Two-tower 5K validation export: `numpy` seed 42 random sample, then sorted.
- Those two 2020-09-16..22 subsets are **not the same people**.

Contaminated validation-week buyers (union of first-N, seed-42 samples, and
cached IDs): **10,610**.

Unused validation-week buyers: **58,374**.

Reserved holdout (seed 20260908, unused only): **8,000** IDs in
`artifacts/overnight/holdout_customer_ids.txt`.

## Selection rule

Choose the simplest configuration whose mean development-fold MAP@12 is within
0.0002 of the best mean. Never use a 2020-09-16..22 number, including the
historical 5K MAP@12 ≈ 0.028, to pick a model.

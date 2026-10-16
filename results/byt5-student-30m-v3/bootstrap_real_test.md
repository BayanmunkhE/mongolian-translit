### pred_real_test.csv (n=195, 2000 paired resamples)

| System | CER↓ | 95% CI | EM↑ | 95% CI |
|---|---:|---|---:|---|
| mns5217 | 23.34 | 22.08–24.66 | 0.00 | 0.00–0.00 |
| naive | 19.66 | 18.42–20.99 | 1.03 | 0.00–2.56 |
| teacher-small-chat | 6.55 | 5.56–7.68 | 27.69 | 21.54–33.85 |
| byt5-student-30m-v3 | 74.27 | 71.62–77.21 | 0.51 | 0.00–1.54 |

| A vs B | CER(A) − CER(B) | 95% CI | A better in |
|---|---:|---|---:|
| mns5217 vs naive | +3.68 | +3.26–+4.10 | 0.0% |
| mns5217 vs teacher-small-chat | +16.79 | +15.59–+18.07 | 0.0% |
| mns5217 vs byt5-student-30m-v3 | -50.93 | -54.10–-47.96 | 100.0% |
| naive vs teacher-small-chat | +13.12 | +11.97–+14.35 | 0.0% |
| naive vs byt5-student-30m-v3 | -54.61 | -57.87–-51.60 | 100.0% |
| teacher-small-chat vs byt5-student-30m-v3 | -67.72 | -70.80–-64.72 | 100.0% |

An interval for the difference that does not include 0 means the gap is unlikely to be chance.

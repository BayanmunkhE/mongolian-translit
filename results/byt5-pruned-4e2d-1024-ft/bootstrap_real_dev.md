### pred_real_dev.csv (n=178, 2000 paired resamples)

| System | CER↓ | 95% CI | EM↑ | 95% CI |
|---|---:|---|---:|---|
| mns5217 | 22.32 | 21.05–23.68 | 0.00 | 0.00–0.00 |
| naive | 18.60 | 17.24–20.01 | 0.56 | 0.00–1.69 |
| teacher-small-chat | 4.65 | 3.72–5.65 | 41.01 | 33.71–48.31 |
| byt5-pruned-4e2d-1024-ft | 7.02 | 5.87–8.24 | 28.09 | 21.35–34.83 |

| A vs B | CER(A) − CER(B) | 95% CI | A better in |
|---|---:|---|---:|
| mns5217 vs naive | +3.72 | +3.23–+4.23 | 0.0% |
| mns5217 vs teacher-small-chat | +17.67 | +16.53–+18.98 | 0.0% |
| mns5217 vs byt5-pruned-4e2d-1024-ft | +15.30 | +14.12–+16.57 | 0.0% |
| naive vs teacher-small-chat | +13.95 | +12.82–+15.09 | 0.0% |
| naive vs byt5-pruned-4e2d-1024-ft | +11.58 | +10.51–+12.71 | 0.0% |
| teacher-small-chat vs byt5-pruned-4e2d-1024-ft | -2.37 | -3.36–-1.45 | 100.0% |

An interval for the difference that does not include 0 means the gap is unlikely to be chance.

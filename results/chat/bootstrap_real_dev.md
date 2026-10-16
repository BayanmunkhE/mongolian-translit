### pred_real_dev.csv (n=178, 2000 paired resamples)

| System | CER↓ | 95% CI | EM↑ | 95% CI |
|---|---:|---|---:|---|
| mns5217 | 22.32 | 21.05–23.68 | 0.00 | 0.00–0.00 |
| naive | 18.60 | 17.24–20.01 | 0.56 | 0.00–1.69 |
| small-v4-real | 4.56 | 3.60–5.53 | 41.01 | 33.71–48.31 |
| small-chat | 4.47 | 3.61–5.39 | 39.33 | 32.02–46.63 |

| A vs B | CER(A) − CER(B) | 95% CI | A better in |
|---|---:|---|---:|
| mns5217 vs naive | +3.72 | +3.23–+4.23 | 0.0% |
| mns5217 vs small-v4-real | +17.76 | +16.61–+19.07 | 0.0% |
| mns5217 vs small-chat | +17.84 | +16.72–+19.14 | 0.0% |
| naive vs small-v4-real | +14.04 | +12.95–+15.23 | 0.0% |
| naive vs small-chat | +14.13 | +12.99–+15.27 | 0.0% |
| small-v4-real vs small-chat | +0.08 | -0.26–+0.40 | 30.3% |

An interval for the difference that does not include 0 means the gap is unlikely to be chance.

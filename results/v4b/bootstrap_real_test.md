### pred_real_test.csv (n=195, 2000 paired resamples)

| System | CER↓ | 95% CI | EM↑ | 95% CI |
|---|---:|---|---:|---|
| small-v4-real | 6.59 | 5.57–7.77 | 30.26 | 23.59–36.41 |
| base-v4-real | 6.19 | 5.22–7.28 | 30.77 | 24.10–36.92 |
| base-v4-real-lr3e-5 | 6.51 | 5.46–7.66 | 29.23 | 23.08–35.38 |
| small-v4-real-1200 | 6.63 | 5.61–7.76 | 29.23 | 23.08–35.38 |

| A vs B | CER(A) − CER(B) | 95% CI | A better in |
|---|---:|---|---:|
| small-v4-real vs base-v4-real | +0.40 | -0.24–+1.14 | 11.0% |
| small-v4-real vs base-v4-real-lr3e-5 | +0.08 | -0.47–+0.67 | 38.2% |
| small-v4-real vs small-v4-real-1200 | -0.04 | -0.27–+0.17 | 64.6% |
| base-v4-real vs base-v4-real-lr3e-5 | -0.32 | -0.76–+0.01 | 96.3% |
| base-v4-real vs small-v4-real-1200 | -0.44 | -1.20–+0.19 | 90.8% |
| base-v4-real-lr3e-5 vs small-v4-real-1200 | -0.12 | -0.72–+0.41 | 66.2% |

An interval for the difference that does not include 0 means the gap is unlikely to be chance.

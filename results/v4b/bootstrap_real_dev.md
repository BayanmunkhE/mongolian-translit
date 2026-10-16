### pred_real_dev.csv (n=178, 2000 paired resamples)

| System | CER↓ | 95% CI | EM↑ | 95% CI |
|---|---:|---|---:|---|
| small-v4-real | 4.60 | 3.63–5.56 | 40.45 | 33.15–47.75 |
| base-v4-real | 4.76 | 3.91–5.67 | 38.76 | 31.46–45.51 |
| base-v4-real-lr3e-5 | 4.93 | 4.01–5.92 | 39.89 | 32.58–46.63 |
| small-v4-real-1200 | 4.53 | 3.59–5.49 | 38.76 | 31.46–46.07 |

| A vs B | CER(A) − CER(B) | 95% CI | A better in |
|---|---:|---|---:|
| small-v4-real vs base-v4-real | -0.17 | -0.60–+0.24 | 78.6% |
| small-v4-real vs base-v4-real-lr3e-5 | -0.33 | -0.73–+0.06 | 94.8% |
| small-v4-real vs small-v4-real-1200 | +0.07 | -0.15–+0.26 | 22.4% |
| base-v4-real vs base-v4-real-lr3e-5 | -0.17 | -0.44–+0.09 | 87.8% |
| base-v4-real vs small-v4-real-1200 | +0.24 | -0.20–+0.67 | 13.2% |
| base-v4-real-lr3e-5 vs small-v4-real-1200 | +0.40 | -0.03–+0.81 | 3.1% |

An interval for the difference that does not include 0 means the gap is unlikely to be chance.

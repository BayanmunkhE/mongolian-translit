### pred_real_dev.csv (n=178, 2000 paired resamples)

| System | CER↓ | 95% CI | EM↑ | 95% CI |
|---|---:|---|---:|---|
| byt5-v3 | 5.45 | 4.41–6.53 | 37.64 | 30.34–44.38 |
| small-v4 | 5.55 | 4.52–6.63 | 32.58 | 25.84–39.33 |
| small-v4-real | 4.60 | 3.63–5.56 | 40.45 | 33.15–47.75 |
| base-v4 | 5.74 | 4.73–6.83 | 32.02 | 25.28–38.76 |
| base-v4-real | 4.76 | 3.91–5.67 | 38.76 | 31.46–45.51 |

| A vs B | CER(A) − CER(B) | 95% CI | A better in |
|---|---:|---|---:|
| byt5-v3 vs small-v4 | -0.10 | -0.61–+0.44 | 65.4% |
| byt5-v3 vs small-v4-real | +0.85 | +0.32–+1.41 | 0.1% |
| byt5-v3 vs base-v4 | -0.29 | -0.88–+0.31 | 83.0% |
| byt5-v3 vs base-v4-real | +0.68 | +0.12–+1.30 | 0.7% |
| small-v4 vs small-v4-real | +0.95 | +0.52–+1.46 | 0.0% |
| small-v4 vs base-v4 | -0.19 | -0.61–+0.20 | 81.0% |
| small-v4 vs base-v4-real | +0.79 | +0.31–+1.29 | 0.1% |
| small-v4-real vs base-v4 | -1.14 | -1.66–-0.66 | 100.0% |
| small-v4-real vs base-v4-real | -0.17 | -0.60–+0.24 | 78.6% |
| base-v4 vs base-v4-real | +0.97 | +0.50–+1.50 | 0.0% |

An interval for the difference that does not include 0 means the gap is unlikely to be chance.

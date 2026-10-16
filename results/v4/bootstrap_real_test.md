### pred_real_test.csv (n=195, 2000 paired resamples)

| System | CER↓ | 95% CI | EM↑ | 95% CI |
|---|---:|---|---:|---|
| byt5-v3 | 7.83 | 6.80–8.90 | 24.62 | 18.97–30.77 |
| small-v4 | 7.50 | 6.47–8.66 | 24.10 | 18.46–30.26 |
| small-v4-real | 6.59 | 5.57–7.77 | 30.26 | 23.59–36.41 |
| base-v4 | 7.13 | 6.09–8.24 | 28.72 | 22.56–34.87 |
| base-v4-real | 6.19 | 5.22–7.28 | 30.77 | 24.10–36.92 |

| A vs B | CER(A) − CER(B) | 95% CI | A better in |
|---|---:|---|---:|
| byt5-v3 vs small-v4 | +0.33 | -0.22–+0.86 | 11.5% |
| byt5-v3 vs small-v4-real | +1.24 | +0.74–+1.76 | 0.0% |
| byt5-v3 vs base-v4 | +0.70 | +0.13–+1.35 | 0.9% |
| byt5-v3 vs base-v4-real | +1.64 | +0.94–+2.42 | 0.0% |
| small-v4 vs small-v4-real | +0.91 | +0.36–+1.55 | 0.1% |
| small-v4 vs base-v4 | +0.37 | -0.31–+1.14 | 14.1% |
| small-v4 vs base-v4-real | +1.31 | +0.55–+2.21 | 0.0% |
| small-v4-real vs base-v4 | -0.54 | -1.13–+0.08 | 95.3% |
| small-v4-real vs base-v4-real | +0.40 | -0.24–+1.14 | 11.0% |
| base-v4 vs base-v4-real | +0.94 | +0.44–+1.47 | 0.0% |

An interval for the difference that does not include 0 means the gap is unlikely to be chance.

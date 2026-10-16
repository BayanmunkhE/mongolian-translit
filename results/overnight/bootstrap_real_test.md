### pred_real_test.csv (n=195, 2000 paired resamples)

| System | CER↓ | 95% CI | EM↑ | 95% CI |
|---|---:|---|---:|---|
| byt5-v3 | 7.83 | 6.80–8.90 | 24.62 | 18.97–30.77 |
| byt5-base | 7.40 | 6.34–8.56 | 23.59 | 17.95–29.74 |
| byt5-v3-warmup | 7.61 | 6.62–8.67 | 22.56 | 16.92–28.72 |
| byt5-v3-adafactor | 7.52 | 6.54–8.60 | 24.10 | 18.46–30.26 |
| byt5-base-20k | 9.17 | 6.91–12.53 | 21.54 | 15.90–27.18 |

| A vs B | CER(A) − CER(B) | 95% CI | A better in |
|---|---:|---|---:|
| byt5-v3 vs byt5-base | +0.43 | -0.13–+1.06 | 7.2% |
| byt5-v3 vs byt5-v3-warmup | +0.22 | -0.26–+0.71 | 17.1% |
| byt5-v3 vs byt5-v3-adafactor | +0.31 | -0.15–+0.82 | 10.1% |
| byt5-v3 vs byt5-base-20k | -1.34 | -4.39–+0.64 | 83.0% |
| byt5-base vs byt5-v3-warmup | -0.22 | -0.76–+0.38 | 76.5% |
| byt5-base vs byt5-v3-adafactor | -0.12 | -0.65–+0.44 | 67.4% |
| byt5-base vs byt5-base-20k | -1.77 | -4.60–+0.04 | 96.9% |
| byt5-v3-warmup vs byt5-v3-adafactor | +0.09 | -0.37–+0.61 | 38.2% |
| byt5-v3-warmup vs byt5-base-20k | -1.56 | -4.60–+0.44 | 88.7% |
| byt5-v3-adafactor vs byt5-base-20k | -1.65 | -4.63–+0.30 | 91.5% |

An interval for the difference that does not include 0 means the gap is unlikely to be chance.

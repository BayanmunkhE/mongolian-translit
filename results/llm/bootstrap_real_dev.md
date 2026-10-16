### pred_real_dev.csv (n=178, 2000 paired resamples)

| System | CER↓ | 95% CI | EM↑ | 95% CI |
|---|---:|---|---:|---|
| small-v4-real | 4.56 | 3.60–5.53 | 41.01 | 33.71–48.31 |
| gemma3-12b-zero | 18.95 | 16.60–21.75 | 5.62 | 2.25–8.99 |
| gemma3-12b-rand5 | 15.59 | 13.89–17.79 | 9.55 | 5.62–14.04 |
| gemma3-12b-rag5 | 12.18 | 10.84–13.71 | 19.10 | 13.48–24.72 |

| A vs B | CER(A) − CER(B) | 95% CI | A better in |
|---|---:|---|---:|
| small-v4-real vs gemma3-12b-zero | -14.40 | -17.01–-12.00 | 100.0% |
| small-v4-real vs gemma3-12b-rand5 | -11.03 | -13.09–-9.37 | 100.0% |
| small-v4-real vs gemma3-12b-rag5 | -7.62 | -9.04–-6.20 | 100.0% |
| gemma3-12b-zero vs gemma3-12b-rand5 | +3.37 | +1.53–+5.49 | 0.0% |
| gemma3-12b-zero vs gemma3-12b-rag5 | +6.77 | +4.90–+8.96 | 0.0% |
| gemma3-12b-rand5 vs gemma3-12b-rag5 | +3.41 | +1.96–+4.99 | 0.0% |

An interval for the difference that does not include 0 means the gap is unlikely to be chance.

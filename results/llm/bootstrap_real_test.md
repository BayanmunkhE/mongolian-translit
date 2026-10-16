### pred_real_test.csv (n=195, 2000 paired resamples)

| System | CER↓ | 95% CI | EM↑ | 95% CI |
|---|---:|---|---:|---|
| small-v4-real | 6.59 | 5.57–7.77 | 30.26 | 23.59–36.41 |
| gemma3-12b-zero | 18.14 | 16.22–20.51 | 7.18 | 3.59–10.77 |
| gemma3-12b-rand5 | 16.88 | 15.38–18.51 | 8.21 | 4.62–12.31 |
| gemma3-12b-rag5 | 12.49 | 11.30–13.59 | 16.41 | 11.79–21.54 |

| A vs B | CER(A) − CER(B) | 95% CI | A better in |
|---|---:|---|---:|
| small-v4-real vs gemma3-12b-zero | -11.55 | -14.07–-9.52 | 100.0% |
| small-v4-real vs gemma3-12b-rand5 | -10.29 | -11.78–-8.72 | 100.0% |
| small-v4-real vs gemma3-12b-rag5 | -5.90 | -6.94–-4.69 | 100.0% |
| gemma3-12b-zero vs gemma3-12b-rand5 | +1.26 | -0.62–+3.79 | 12.3% |
| gemma3-12b-zero vs gemma3-12b-rag5 | +5.65 | +3.99–+7.85 | 0.0% |
| gemma3-12b-rand5 vs gemma3-12b-rag5 | +4.39 | +3.17–+5.75 | 0.0% |

An interval for the difference that does not include 0 means the gap is unlikely to be chance.

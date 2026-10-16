### pred_real_test.csv (n=195, 2000 paired resamples)

| System | CER↓ | 95% CI | EM↑ | 95% CI |
|---|---:|---|---:|---|
| mns5217 | 23.08 | 21.88–24.33 | 0.00 | 0.00–0.00 |
| naive | 19.33 | 18.16–20.60 | 1.03 | 0.00–2.56 |
| byt5-v2 | 8.61 | 7.46–9.83 | 20.00 | 14.36–25.64 |
| byt5-v3 | 8.34 | 7.26–9.56 | 24.10 | 18.46–30.26 |

| A vs B | CER(A) − CER(B) | 95% CI | A better in |
|---|---:|---|---:|
| mns5217 vs naive | +3.75 | +3.33–+4.18 | 0.0% |
| mns5217 vs byt5-v2 | +14.47 | +13.25–+15.69 | 0.0% |
| mns5217 vs byt5-v3 | +14.74 | +13.50–+15.92 | 0.0% |
| naive vs byt5-v2 | +10.72 | +9.61–+11.85 | 0.0% |
| naive vs byt5-v3 | +10.99 | +9.89–+12.06 | 0.0% |
| byt5-v2 vs byt5-v3 | +0.27 | -0.34–+0.79 | 16.4% |

An interval for the difference that does not include 0 means the gap is unlikely to be chance.

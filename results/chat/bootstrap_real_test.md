### pred_real_test.csv (n=195, 2000 paired resamples)

| System | CER↓ | 95% CI | EM↑ | 95% CI |
|---|---:|---|---:|---|
| mns5217 | 23.34 | 22.08–24.66 | 0.00 | 0.00–0.00 |
| naive | 19.66 | 18.42–20.99 | 1.03 | 0.00–2.56 |
| small-v4-real | 6.59 | 5.57–7.77 | 30.26 | 23.59–36.41 |
| small-chat | 6.53 | 5.57–7.62 | 27.18 | 21.03–33.33 |

| A vs B | CER(A) − CER(B) | 95% CI | A better in |
|---|---:|---|---:|
| mns5217 vs naive | +3.68 | +3.26–+4.10 | 0.0% |
| mns5217 vs small-v4-real | +16.75 | +15.52–+18.02 | 0.0% |
| mns5217 vs small-chat | +16.82 | +15.65–+18.05 | 0.0% |
| naive vs small-v4-real | +13.08 | +11.86–+14.28 | 0.0% |
| naive vs small-chat | +13.14 | +12.02–+14.31 | 0.0% |
| small-v4-real vs small-chat | +0.06 | -0.31–+0.55 | 41.1% |

An interval for the difference that does not include 0 means the gap is unlikely to be chance.

### pred_real_test.csv (n=195, 2000 paired resamples)

| System | CER↓ | 95% CI | EM↑ | 95% CI |
|---|---:|---|---:|---|
| mns5217 | 23.34 | 22.08–24.66 | 0.00 | 0.00–0.00 |
| naive | 19.66 | 18.42–20.99 | 1.03 | 0.00–2.56 |
| small-chat | 6.53 | 5.57–7.62 | 27.18 | 21.03–33.33 |
| small-chat2 | 6.15 | 5.21–7.23 | 28.21 | 22.05–34.36 |

| A vs B | CER(A) − CER(B) | 95% CI | A better in |
|---|---:|---|---:|
| mns5217 vs naive | +3.68 | +3.26–+4.10 | 0.0% |
| mns5217 vs small-chat | +16.82 | +15.65–+18.05 | 0.0% |
| mns5217 vs small-chat2 | +17.19 | +15.95–+18.52 | 0.0% |
| naive vs small-chat | +13.14 | +12.02–+14.31 | 0.0% |
| naive vs small-chat2 | +13.52 | +12.29–+14.76 | 0.0% |
| small-chat vs small-chat2 | +0.38 | -0.06–+0.89 | 4.0% |

An interval for the difference that does not include 0 means the gap is unlikely to be chance.

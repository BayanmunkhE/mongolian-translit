### pred_chat_test.csv (n=40, 2000 paired resamples)

| System | CER↓ | 95% CI | EM↑ | 95% CI |
|---|---:|---|---:|---|
| mns5217 | 31.47 | 26.38–36.37 | 0.00 | 0.00–0.00 |
| naive | 30.59 | 25.57–35.20 | 0.00 | 0.00–0.00 |
| small-v4-real | 9.76 | 6.93–12.48 | 32.50 | 17.50–47.50 |
| small-chat | 6.58 | 3.78–9.63 | 55.00 | 40.00–70.00 |

| A vs B | CER(A) − CER(B) | 95% CI | A better in |
|---|---:|---|---:|
| mns5217 vs naive | +0.88 | -0.22–+1.98 | 4.9% |
| mns5217 vs small-v4-real | +21.71 | +16.49–+26.88 | 0.0% |
| mns5217 vs small-chat | +24.89 | +19.64–+30.25 | 0.0% |
| naive vs small-v4-real | +20.83 | +15.72–+25.66 | 0.0% |
| naive vs small-chat | +24.01 | +19.08–+29.03 | 0.0% |
| small-v4-real vs small-chat | +3.18 | +1.07–+5.34 | 0.2% |

An interval for the difference that does not include 0 means the gap is unlikely to be chance.

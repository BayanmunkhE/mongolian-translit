### pred_chat_test.csv (n=40, 2000 paired resamples)

| System | CER↓ | 95% CI | EM↑ | 95% CI |
|---|---:|---|---:|---|
| mns5217 | 31.47 | 26.38–36.37 | 0.00 | 0.00–0.00 |
| naive | 30.59 | 25.57–35.20 | 0.00 | 0.00–0.00 |
| small-chat | 6.58 | 3.78–9.63 | 55.00 | 40.00–70.00 |
| small-chat2 | 5.26 | 2.42–8.05 | 72.50 | 57.50–85.00 |

| A vs B | CER(A) − CER(B) | 95% CI | A better in |
|---|---:|---|---:|
| mns5217 vs naive | +0.88 | -0.22–+1.98 | 4.9% |
| mns5217 vs small-chat | +24.89 | +19.64–+30.25 | 0.0% |
| mns5217 vs small-chat2 | +26.21 | +21.18–+31.53 | 0.0% |
| naive vs small-chat | +24.01 | +19.08–+29.03 | 0.0% |
| naive vs small-chat2 | +25.33 | +20.35–+30.26 | 0.0% |
| small-chat vs small-chat2 | +1.32 | -0.41–+3.23 | 5.8% |

An interval for the difference that does not include 0 means the gap is unlikely to be chance.

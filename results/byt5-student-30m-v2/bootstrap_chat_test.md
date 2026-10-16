### pred_chat_test.csv (n=40, 2000 paired resamples)

| System | CER↓ | 95% CI | EM↑ | 95% CI |
|---|---:|---|---:|---|
| mns5217 | 31.47 | 26.38–36.37 | 0.00 | 0.00–0.00 |
| naive | 30.59 | 25.57–35.20 | 0.00 | 0.00–0.00 |
| teacher-small-chat | 6.36 | 3.93–8.85 | 50.00 | 35.00–65.00 |
| byt5-student-30m-v2 | 84.10 | 76.63–90.81 | 2.50 | 0.00–7.50 |

| A vs B | CER(A) − CER(B) | 95% CI | A better in |
|---|---:|---|---:|
| mns5217 vs naive | +0.88 | -0.22–+1.98 | 4.9% |
| mns5217 vs teacher-small-chat | +25.11 | +19.97–+30.27 | 0.0% |
| mns5217 vs byt5-student-30m-v2 | -52.63 | -59.32–-45.73 | 100.0% |
| naive vs teacher-small-chat | +24.23 | +19.41–+29.07 | 0.0% |
| naive vs byt5-student-30m-v2 | -53.51 | -60.08–-46.78 | 100.0% |
| teacher-small-chat vs byt5-student-30m-v2 | -77.74 | -84.22–-70.57 | 100.0% |

An interval for the difference that does not include 0 means the gap is unlikely to be chance.

### pred_chat_test.csv (n=40, 2000 paired resamples)

| System | CER↓ | 95% CI | EM↑ | 95% CI |
|---|---:|---|---:|---|
| mns5217 | 31.47 | 26.38–36.37 | 0.00 | 0.00–0.00 |
| naive | 30.59 | 25.57–35.20 | 0.00 | 0.00–0.00 |
| teacher-small-chat | 6.36 | 3.93–8.85 | 50.00 | 35.00–65.00 |
| byt5-pruned-4e2d-1024-ft | 9.10 | 5.71–12.50 | 42.50 | 27.50–57.50 |

| A vs B | CER(A) − CER(B) | 95% CI | A better in |
|---|---:|---|---:|
| mns5217 vs naive | +0.88 | -0.22–+1.98 | 4.9% |
| mns5217 vs teacher-small-chat | +25.11 | +19.97–+30.27 | 0.0% |
| mns5217 vs byt5-pruned-4e2d-1024-ft | +22.37 | +17.93–+26.69 | 0.0% |
| naive vs teacher-small-chat | +24.23 | +19.41–+29.07 | 0.0% |
| naive vs byt5-pruned-4e2d-1024-ft | +21.49 | +17.37–+25.43 | 0.0% |
| teacher-small-chat vs byt5-pruned-4e2d-1024-ft | -2.74 | -6.30–+0.48 | 94.7% |

An interval for the difference that does not include 0 means the gap is unlikely to be chance.

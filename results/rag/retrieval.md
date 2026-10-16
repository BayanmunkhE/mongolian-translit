### real_dev, bm25 (n=178 of 178 queries with gold results, k=10)

| Query | overlap@k ↑ | 95% CI | top-1 agreement ↑ | 95% CI |
|---|---:|---|---:|---|
| latin | 4.3 | 2.7–6.3 | 2.2 | 0.6–4.5 |
| naive | 21.6 | 17.8–25.4 | 12.9 | 7.9–18.0 |
| mns5217 | 18.9 | 15.3–22.6 | 10.7 | 6.2–15.2 |
| byt5-v3 | 76.3 | 71.9–80.2 | 67.4 | 60.7–74.2 |
| small-v4-real | 79.1 | 74.8–82.9 | 71.3 | 64.6–78.1 |
| base-v4-real | 79.9 | 75.8–83.6 | 71.3 | 64.6–77.5 |

| A vs B | overlap(A) − overlap(B) | 95% CI |
|---|---:|---|
| mns5217 vs byt5-v3 | -57.5 | -62.5–-52.4 |
| mns5217 vs small-v4-real | -60.2 | -65.1–-55.3 |
| mns5217 vs base-v4-real | -61.0 | -65.7–-56.2 |
| byt5-v3 vs small-v4-real | -2.8 | -5.3–-0.2 |
| byt5-v3 vs base-v4-real | -3.5 | -6.1–-1.0 |
| small-v4-real vs base-v4-real | -0.8 | -2.5–+0.8 |

### real_dev, e5 (n=178 of 178 queries with gold results, k=10)

| Query | overlap@k ↑ | 95% CI | top-1 agreement ↑ | 95% CI |
|---|---:|---|---:|---|
| latin | 8.0 | 6.5–9.6 | 2.2 | 0.6–4.5 |
| naive | 39.9 | 36.5–43.1 | 26.4 | 20.2–33.7 |
| mns5217 | 34.0 | 31.0–37.2 | 17.4 | 12.4–23.0 |
| byt5-v3 | 78.0 | 74.2–81.6 | 79.8 | 74.2–85.4 |
| small-v4-real | 83.4 | 79.9–86.5 | 84.8 | 79.2–89.9 |
| base-v4-real | 82.1 | 78.8–85.2 | 83.1 | 77.5–88.2 |

| A vs B | overlap(A) − overlap(B) | 95% CI |
|---|---:|---|
| mns5217 vs byt5-v3 | -44.0 | -48.4–-39.3 |
| mns5217 vs small-v4-real | -49.3 | -53.8–-44.6 |
| mns5217 vs base-v4-real | -48.1 | -52.1–-43.9 |
| byt5-v3 vs small-v4-real | -5.3 | -8.1–-2.8 |
| byt5-v3 vs base-v4-real | -4.1 | -6.9–-1.2 |
| small-v4-real vs base-v4-real | +1.2 | -1.2–+3.7 |

### real_test, bm25 (n=195 of 195 queries with gold results, k=10)

| Query | overlap@k ↑ | 95% CI | top-1 agreement ↑ | 95% CI |
|---|---:|---|---:|---|
| latin | 6.5 | 4.3–8.8 | 5.1 | 2.1–8.7 |
| naive | 20.5 | 17.1–23.9 | 13.3 | 8.7–18.5 |
| mns5217 | 18.3 | 15.1–21.7 | 11.3 | 7.2–15.9 |
| byt5-v3 | 70.6 | 65.6–75.1 | 65.6 | 59.0–72.3 |
| small-v4-real | 75.0 | 70.4–79.3 | 71.8 | 65.1–77.9 |
| base-v4-real | 74.9 | 70.4–79.1 | 72.8 | 66.7–78.5 |

| A vs B | overlap(A) − overlap(B) | 95% CI |
|---|---:|---|
| mns5217 vs byt5-v3 | -52.3 | -57.1–-47.3 |
| mns5217 vs small-v4-real | -56.7 | -61.5–-51.9 |
| mns5217 vs base-v4-real | -56.6 | -61.1–-51.9 |
| byt5-v3 vs small-v4-real | -4.5 | -7.6–-1.6 |
| byt5-v3 vs base-v4-real | -4.4 | -7.7–-1.3 |
| small-v4-real vs base-v4-real | +0.1 | -2.1–+2.3 |

### real_test, e5 (n=195 of 195 queries with gold results, k=10)

| Query | overlap@k ↑ | 95% CI | top-1 agreement ↑ | 95% CI |
|---|---:|---|---:|---|
| latin | 9.7 | 8.0–11.6 | 7.2 | 3.6–11.3 |
| naive | 40.4 | 37.1–43.7 | 29.7 | 23.1–36.9 |
| mns5217 | 34.2 | 31.2–37.5 | 24.1 | 18.5–30.8 |
| byt5-v3 | 77.9 | 74.6–81.1 | 74.9 | 68.7–81.0 |
| small-v4-real | 81.4 | 78.5–84.4 | 76.4 | 70.3–82.6 |
| base-v4-real | 81.3 | 78.5–84.1 | 77.4 | 71.3–83.1 |

| A vs B | overlap(A) − overlap(B) | 95% CI |
|---|---:|---|
| mns5217 vs byt5-v3 | -43.7 | -47.6–-39.7 |
| mns5217 vs small-v4-real | -47.2 | -51.2–-43.3 |
| mns5217 vs base-v4-real | -47.1 | -50.8–-43.1 |
| byt5-v3 vs small-v4-real | -3.5 | -5.7–-1.5 |
| byt5-v3 vs base-v4-real | -3.3 | -5.6–-1.1 |
| small-v4-real vs base-v4-real | +0.2 | -1.7–+1.9 |

overlap@k: share of the gold (hand-made Cyrillic) query's top-k passages that the query version also retrieves. Gold vs itself is 100 by definition.

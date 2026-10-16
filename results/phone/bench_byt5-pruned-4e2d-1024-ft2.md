# Phone-constraint benchmark: models/byt5-pruned-4e2d-1024-ft2

CPU 4 threads, one message at a time, beam = decoding. Weights on disk: 175 MB fp32; int8 Linear layers: 50 MB.

### real_test (n=195)

| Config | CER↓ | EM↑ | p50 ms | p95 ms | mean ms |
|---|---:|---:|---:|---:|---:|
| fp32 beam4 | 8.12 | 28.2 | 1381 | 9117 | 2658 |
| fp32 greedy | 7.72 | 28.2 | 478 | 2694 | 812 |
| int8 greedy | 7.89 | 28.2 | 349 | 1578 | 506 |

### chat_test (n=40)

| Config | CER↓ | EM↑ | p50 ms | p95 ms | mean ms |
|---|---:|---:|---:|---:|---:|
| fp32 beam4 | 7.35 | 65.0 | 428 | 1979 | 588 |
| fp32 greedy | 7.46 | 65.0 | 179 | 645 | 226 |
| int8 greedy | 7.35 | 65.0 | 145 | 412 | 167 |

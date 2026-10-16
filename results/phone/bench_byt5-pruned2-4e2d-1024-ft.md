# Phone-constraint benchmark: models/byt5-pruned2-4e2d-1024-ft

CPU 4 threads, one message at a time, beam = decoding. Weights on disk: 175 MB fp32; int8 Linear layers: 50 MB.

### real_test (n=195)

| Config | CER↓ | EM↑ | p50 ms | p95 ms | mean ms |
|---|---:|---:|---:|---:|---:|
| fp32 beam4 | 8.24 | 26.2 | 1384 | 9330 | 2671 |
| fp32 greedy | 8.34 | 26.2 | 465 | 2701 | 828 |
| int8 greedy | 8.53 | 26.2 | 344 | 1567 | 509 |

### chat_test (n=40)

| Config | CER↓ | EM↑ | p50 ms | p95 ms | mean ms |
|---|---:|---:|---:|---:|---:|
| fp32 beam4 | 8.11 | 60.0 | 424 | 1941 | 585 |
| fp32 greedy | 7.79 | 60.0 | 179 | 644 | 224 |
| int8 greedy | 7.79 | 60.0 | 145 | 402 | 165 |

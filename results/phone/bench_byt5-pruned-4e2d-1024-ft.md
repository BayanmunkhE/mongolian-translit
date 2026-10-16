# Phone-constraint benchmark: models/byt5-pruned-4e2d-1024-ft

CPU 4 threads, one message at a time, beam = decoding. Weights on disk: 175 MB fp32; int8 Linear layers: 50 MB.

### real_test (n=195)

| Config | CER↓ | EM↑ | p50 ms | p95 ms | mean ms |
|---|---:|---:|---:|---:|---:|
| fp32 beam4 | 9.02 | 25.1 | 2036 | 13193 | 4107 |
| fp32 greedy | 9.02 | 26.2 | 822 | 4288 | 1406 |
| int8 greedy | 9.16 | 25.6 | 526 | 2430 | 780 |

### chat_test (n=40)

| Config | CER↓ | EM↑ | p50 ms | p95 ms | mean ms |
|---|---:|---:|---:|---:|---:|
| fp32 beam4 | 9.43 | 42.5 | 719 | 3291 | 993 |
| fp32 greedy | 9.1 | 42.5 | 314 | 1120 | 392 |
| int8 greedy | 9.1 | 45.0 | 198 | 592 | 223 |

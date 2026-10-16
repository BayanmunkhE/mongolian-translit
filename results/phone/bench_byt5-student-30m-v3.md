# Phone-constraint benchmark: models/byt5-student-30m-v3

CPU 4 threads, one message at a time, beam = decoding. Weights on disk: 113 MB fp32; int8 Linear layers: 30 MB.

### real_test (n=195)

| Config | CER↓ | EM↑ | p50 ms | p95 ms | mean ms |
|---|---:|---:|---:|---:|---:|
| fp32 beam4 | 70.67 | 1.0 | 128 | 241 | 140 |
| fp32 greedy | 72.78 | 0.5 | 106 | 200 | 114 |
| int8 greedy | 72.45 | 0.5 | 117 | 232 | 132 |

### chat_test (n=40)

| Config | CER↓ | EM↑ | p50 ms | p95 ms | mean ms |
|---|---:|---:|---:|---:|---:|
| fp32 beam4 | 71.49 | 0.0 | 91 | 182 | 95 |
| fp32 greedy | 81.69 | 0.0 | 74 | 155 | 84 |
| int8 greedy | 79.5 | 0.0 | 80 | 197 | 99 |

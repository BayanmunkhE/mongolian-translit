# Phone-constraint benchmark: models/byt5-student-30m

CPU 4 threads, one message at a time, beam = decoding. Weights on disk: 113 MB fp32; int8 Linear layers: 29 MB.

### real_test (n=195)

| Config | CER↓ | EM↑ | p50 ms | p95 ms | mean ms |
|---|---:|---:|---:|---:|---:|
| fp32 beam4 | 71.94 | 1.5 | 187 | 370 | 206 |
| fp32 greedy | 73.15 | 0.0 | 135 | 279 | 154 |
| int8 greedy | 73.15 | 0.0 | 156 | 337 | 180 |

### chat_test (n=40)

| Config | CER↓ | EM↑ | p50 ms | p95 ms | mean ms |
|---|---:|---:|---:|---:|---:|
| fp32 beam4 | 82.57 | 0.0 | 142 | 334 | 172 |
| fp32 greedy | 92.54 | 0.0 | 119 | 214 | 121 |
| int8 greedy | 92.32 | 0.0 | 117 | 321 | 142 |

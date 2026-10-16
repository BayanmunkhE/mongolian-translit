# Phone-constraint benchmark: models/byt5-student-30m-v2

CPU 4 threads, one message at a time, beam = decoding. Weights on disk: 113 MB fp32; int8 Linear layers: 30 MB.

### real_test (n=195)

| Config | CER↓ | EM↑ | p50 ms | p95 ms | mean ms |
|---|---:|---:|---:|---:|---:|
| fp32 beam4 | 72.58 | 0.5 | 110 | 189 | 125 |
| fp32 greedy | 72.95 | 0.0 | 91 | 184 | 105 |
| int8 greedy | 73.89 | 0.0 | 113 | 216 | 129 |

### chat_test (n=40)

| Config | CER↓ | EM↑ | p50 ms | p95 ms | mean ms |
|---|---:|---:|---:|---:|---:|
| fp32 beam4 | 67.11 | 2.5 | 87 | 157 | 90 |
| fp32 greedy | 82.46 | 2.5 | 68 | 201 | 81 |
| int8 greedy | 79.61 | 2.5 | 79 | 205 | 96 |

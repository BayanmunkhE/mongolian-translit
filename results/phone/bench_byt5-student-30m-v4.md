# Phone-constraint benchmark: models/byt5-student-30m-v4

CPU 4 threads, one message at a time, beam = decoding. Weights on disk: 115 MB fp32; int8 Linear layers: 31 MB.

### real_test (n=195)

| Config | CER↓ | EM↑ | p50 ms | p95 ms | mean ms |
|---|---:|---:|---:|---:|---:|
| fp32 beam4 | 67.42 | 2.1 | 134 | 253 | 149 |
| fp32 greedy | 70.12 | 2.1 | 105 | 245 | 119 |
| int8 greedy | 70.28 | 1.5 | 126 | 293 | 146 |

### chat_test (n=40)

| Config | CER↓ | EM↑ | p50 ms | p95 ms | mean ms |
|---|---:|---:|---:|---:|---:|
| fp32 beam4 | 74.01 | 2.5 | 87 | 169 | 92 |
| fp32 greedy | 81.25 | 2.5 | 72 | 175 | 79 |
| int8 greedy | 77.19 | 2.5 | 88 | 178 | 92 |

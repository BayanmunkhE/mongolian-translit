# Phone-constraint benchmark: models/byt5-pruned-4e1d-768-ft

CPU 4 threads, one message at a time, beam = decoding. Weights on disk: 119 MB fp32; int8 Linear layers: 32 MB.

### real_test (n=195)

| Config | CER↓ | EM↑ | p50 ms | p95 ms | mean ms |
|---|---:|---:|---:|---:|---:|
| fp32 beam4 | 11.49 | 21.0 | 797 | 5167 | 1676 |
| fp32 greedy | 11.54 | 20.0 | 318 | 1628 | 572 |
| int8 greedy | 37.87 | 3.6 | 195 | 1041 | 324 |

### chat_test (n=40)

| Config | CER↓ | EM↑ | p50 ms | p95 ms | mean ms |
|---|---:|---:|---:|---:|---:|
| fp32 beam4 | 11.62 | 32.5 | 344 | 1581 | 474 |
| fp32 greedy | 12.5 | 30.0 | 99 | 324 | 119 |
| int8 greedy | 20.94 | 12.5 | 115 | 306 | 125 |

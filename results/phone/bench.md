# Phone-constraint benchmark: models/byt5-small-chat

CPU 4 threads, one message at a time, beam = decoding. Weights on disk: 1143 MB fp32; int8 Linear layers: 288 MB.

### real_test (n=195)

| Config | CER↓ | EM↑ | p50 ms | p95 ms | mean ms |
|---|---:|---:|---:|---:|---:|
| fp32 beam4 | 6.53 | 27.2 | 1228 | 3407 | 1471 |
| fp32 greedy | 6.55 | 27.7 | 1036 | 2718 | 1227 |
| int8 greedy | 6.91 | 25.6 | 443 | 1178 | 515 |

### chat_test (n=40)

| Config | CER↓ | EM↑ | p50 ms | p95 ms | mean ms |
|---|---:|---:|---:|---:|---:|
| fp32 beam4 | 6.58 | 55.0 | 671 | 1474 | 698 |
| fp32 greedy | 6.36 | 50.0 | 571 | 1292 | 609 |
| int8 greedy | 6.8 | 50.0 | 253 | 567 | 258 |

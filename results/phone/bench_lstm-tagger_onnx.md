# Tagger ONNX benchmark: models/lstm-tagger

- int8=False chat_test: CER 6.58 EM 65.0 | 1 ms/message
- int8=False real_test: CER 8.52 EM 23.1 | 3 ms/message
- int8=False single words: p50 0.5 ms, p95 0.9 ms
- int8=True chat_test: CER 6.69 EM 65.0 | 1 ms/message
- int8=True real_test: CER 8.52 EM 23.1 | 2 ms/message
- int8=True single words: p50 0.4 ms, p95 0.5 ms
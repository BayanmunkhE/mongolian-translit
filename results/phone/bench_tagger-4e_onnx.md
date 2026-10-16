# Tagger ONNX benchmark: models/tagger-4e

- int8=False chat_test: CER 8.55 EM 50.0 | 4 ms/message
- int8=False real_test: CER 9.17 EM 22.1 | 7 ms/message
- int8=False single words: p50 1.4 ms, p95 1.9 ms
- int8=True chat_test: CER 8.44 EM 45.0 | 1 ms/message
- int8=True real_test: CER 9.18 EM 22.1 | 2 ms/message
- int8=True single words: p50 0.5 ms, p95 0.7 ms
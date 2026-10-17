# Tagger ONNX benchmark: models/tagger-4e-full

- int8=False chat_test: CER 6.69 EM 62.5 | 4 ms/message
- int8=False real_test: CER 8.85 EM 22.1 | 7 ms/message
- int8=False single words: p50 1.4 ms, p95 1.9 ms
- int8=True chat_test: CER 6.58 EM 62.5 | 1 ms/message
- int8=True real_test: CER 8.77 EM 22.6 | 2 ms/message
- int8=True single words: p50 0.5 ms, p95 0.7 ms
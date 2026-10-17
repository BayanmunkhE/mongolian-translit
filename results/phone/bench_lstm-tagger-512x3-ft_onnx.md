# Tagger ONNX benchmark: models/lstm-tagger-512x3-ft

- int8=False chat_test: CER 5.59 EM 67.5 | 3 ms/message
- int8=False real_test: CER 7.91 EM 24.1 | 6 ms/message
- int8=False single words: p50 0.8 ms, p95 1.2 ms
- int8=True chat_test: CER 5.59 EM 67.5 | 1 ms/message
- int8=True real_test: CER 7.92 EM 24.6 | 2 ms/message
- int8=True single words: p50 0.3 ms, p95 0.5 ms
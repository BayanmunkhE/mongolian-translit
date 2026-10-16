# LLM transliteration

Last run: ollama gemma3:12b, effort -, k=5. Tables include systems merged from earlier runs on the same rows.

### real_dev (n=20)

| System | CER↓ | WER↓ | EM↑ | CER (ci)↓ | EM (ci)↑ | Ambig↑ | ө/о | ү/у | х | и/й/ь/ъ | ш/щ | Eng↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| small-v4-real | 4.19 | 11.84 | 50.00 | 4.11 | 50.00 | 96.52 | 97.37 | 93.20 | 100.00 | 97.44 | 100.00 | 71.43 |
| base-v4-real | 5.45 | 14.04 | 45.00 | 5.29 | 45.00 | 95.25 | 92.11 | 93.20 | 100.00 | 97.44 | 100.00 | 71.43 |
| gemma3-12b-zero | 21.56 | 46.49 | 10.00 | 16.51 | 10.00 | 78.48 | 76.32 | 70.87 | 89.58 | 80.77 | 100.00 | 42.86 |
| gemma3-12b-rand5 | 16.11 | 39.91 | 5.00 | 15.56 | 10.00 | 76.27 | 72.37 | 68.93 | 91.67 | 76.92 | 100.00 | 57.14 |
| gemma3-12b-rag5 | 12.72 | 32.46 | 30.00 | 12.32 | 30.00 | 82.91 | 71.05 | 81.55 | 91.67 | 88.46 | 100.00 | 57.14 |

All numbers in %. ci = case-insensitive. Ambig = accuracy on ambiguous letters (ө/о, ү/у, х, и/й/ь/ъ, ш/щ). Eng = Latin words preserved (7 in references).


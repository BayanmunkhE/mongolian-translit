# Evaluation results

### synthetic_test (n=2000)

| System | CER↓ | WER↓ | EM↑ | CER (ci)↓ | EM (ci)↑ | Ambig↑ | ө/о | ү/у | х | и/й/ь/ъ | ш/щ | Eng↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| byt5-v3-warmup | 2.15 | 8.80 | 44.15 | 1.48 | 58.15 | 97.54 | 96.98 | 96.67 | 99.44 | 97.52 | 99.38 | 90.13 |

All numbers in %. ci = case-insensitive. Ambig = accuracy on ambiguous letters (ө/о, ү/у, х, и/й/ь/ъ, ш/щ). Eng = Latin words preserved (790 in references).

### real_test (n=195)

| System | CER↓ | WER↓ | EM↑ | CER (ci)↓ | EM (ci)↑ | Ambig↑ | ө/о | ү/у | х | и/й/ь/ъ | ш/щ | Eng↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| byt5-v3-warmup | 7.61 | 23.97 | 22.56 | 6.84 | 31.28 | 94.48 | 95.79 | 94.09 | 96.89 | 91.60 | 98.80 | 56.52 |

All numbers in %. ci = case-insensitive. Ambig = accuracy on ambiguous letters (ө/о, ү/у, х, и/й/ь/ъ, ш/щ). Eng = Latin words preserved (46 in references).

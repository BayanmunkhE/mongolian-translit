# Evaluation results

### synthetic_test (n=2000)

| System | CER↓ | WER↓ | EM↑ | CER (ci)↓ | EM (ci)↑ | Ambig↑ | ө/о | ү/у | х | и/й/ь/ъ | ш/щ | Eng↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v3-beam4 | 2.15 | 8.72 | 43.90 | 1.49 | 57.80 | 97.58 | 97.22 | 96.69 | 99.41 | 97.43 | 99.59 | 90.38 |

All numbers in %. ci = case-insensitive. Ambig = accuracy on ambiguous letters (ө/о, ү/у, х, и/й/ь/ъ, ш/щ). Eng = Latin words preserved (790 in references).

### real_test (n=195)

| System | CER↓ | WER↓ | EM↑ | CER (ci)↓ | EM (ci)↑ | Ambig↑ | ө/о | ү/у | х | и/й/ь/ъ | ш/щ | Eng↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v3-beam4 | 8.34 | 24.26 | 24.10 | 7.56 | 32.82 | 94.22 | 94.74 | 95.08 | 96.89 | 90.88 | 97.59 | 46.15 |

All numbers in %. ci = case-insensitive. Ambig = accuracy on ambiguous letters (ө/о, ү/у, х, и/й/ь/ъ, ш/щ). Eng = Latin words preserved (39 in references).

# Evaluation results

### synthetic_test (n=2000)

| System | CER↓ | WER↓ | EM↑ | CER (ci)↓ | EM (ci)↑ | Ambig↑ | ө/о | ү/у | х | и/й/ь/ъ | ш/щ | Eng↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| byt5-base | 1.88 | 7.66 | 48.50 | 1.24 | 62.70 | 97.99 | 97.63 | 97.31 | 99.39 | 97.94 | 99.48 | 91.52 |

All numbers in %. ci = case-insensitive. Ambig = accuracy on ambiguous letters (ө/о, ү/у, х, и/й/ь/ъ, ш/щ). Eng = Latin words preserved (790 in references).

### real_test (n=195)

| System | CER↓ | WER↓ | EM↑ | CER (ci)↓ | EM (ci)↑ | Ambig↑ | ө/о | ү/у | х | и/й/ь/ъ | ш/щ | Eng↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| byt5-base | 7.40 | 22.64 | 23.59 | 6.59 | 33.85 | 94.94 | 95.94 | 94.39 | 98.02 | 92.23 | 98.80 | 65.22 |

All numbers in %. ci = case-insensitive. Ambig = accuracy on ambiguous letters (ө/о, ү/у, х, и/й/ь/ъ, ш/щ). Eng = Latin words preserved (46 in references).

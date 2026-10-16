# Evaluation results

### synthetic_test (n=2000)

| System | CER↓ | WER↓ | EM↑ | CER (ci)↓ | EM (ci)↑ | Ambig↑ | ө/о | ү/у | х | и/й/ь/ъ | ш/щ | Eng↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v3-beam8 | 2.14 | 8.67 | 44.10 | 1.48 | 57.95 | 97.59 | 97.21 | 96.64 | 99.41 | 97.53 | 99.59 | 90.38 |

All numbers in %. ci = case-insensitive. Ambig = accuracy on ambiguous letters (ө/о, ү/у, х, и/й/ь/ъ, ш/щ). Eng = Latin words preserved (790 in references).

### real_test (n=195)

| System | CER↓ | WER↓ | EM↑ | CER (ci)↓ | EM (ci)↑ | Ambig↑ | ө/о | ү/у | х | и/й/ь/ъ | ш/щ | Eng↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v3-beam8 | 8.30 | 24.26 | 24.10 | 7.52 | 32.82 | 94.22 | 94.74 | 95.08 | 96.89 | 90.88 | 97.59 | 46.15 |

All numbers in %. ci = case-insensitive. Ambig = accuracy on ambiguous letters (ө/о, ү/у, х, и/й/ь/ъ, ш/щ). Eng = Latin words preserved (39 in references).

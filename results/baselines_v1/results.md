# Evaluation results

### synthetic_test (n=2000)

| System | CER↓ | WER↓ | EM↑ | CER (ci)↓ | EM (ci)↑ | Ambig↑ | ө/о | ү/у | х | и/й/ь/ъ | ш/щ | Eng↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mns5217 | 19.84 | 67.27 | 0.45 | 17.87 | 0.80 | 57.11 | 74.31 | 60.58 | 27.00 | 49.95 | 98.10 | 0.90 |
| naive | 17.76 | 62.65 | 0.40 | 15.68 | 0.60 | 72.67 | 74.31 | 75.39 | 99.45 | 56.21 | 98.10 | 0.00 |

All numbers in %. ci = case-insensitive. Ambig = accuracy on ambiguous letters (ө/о, ү/у, х, и/й/ь/ъ, ш/щ). Eng = Latin words preserved (892 in references).

### real_test (n=195)

| System | CER↓ | WER↓ | EM↑ | CER (ci)↓ | EM (ci)↑ | Ambig↑ | ө/о | ү/у | х | и/й/ь/ъ | ш/щ | Eng↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mns5217 | 23.08 | 63.33 | 0.00 | 22.95 | 0.00 | 48.05 | 72.22 | 57.23 | 1.13 | 32.70 | 100.00 | 0.00 |
| naive | 19.33 | 57.04 | 1.03 | 19.20 | 1.03 | 63.75 | 72.22 | 62.00 | 98.02 | 32.86 | 100.00 | 0.00 |

All numbers in %. ci = case-insensitive. Ambig = accuracy on ambiguous letters (ө/о, ү/у, х, и/й/ь/ъ, ш/щ). Eng = Latin words preserved (39 in references).

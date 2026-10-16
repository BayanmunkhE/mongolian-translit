# Evaluation results

### synthetic_test (n=2000)

| System | CER↓ | WER↓ | EM↑ | CER (ci)↓ | EM (ci)↑ | Ambig↑ | ө/о | ү/у | х | и/й/ь/ъ | ш/щ | Eng↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| byt5-v3-lr5e-4 | 2.00 | 8.03 | 47.65 | 1.25 | 61.95 | 97.92 | 97.59 | 97.26 | 99.38 | 97.84 | 99.01 | 96.44 |

All numbers in %. ci = case-insensitive. Ambig = accuracy on ambiguous letters (ө/о, ү/у, х, и/й/ь/ъ, ш/щ). Eng = Latin words preserved (730 in references).

### real_test (n=195)

| System | CER↓ | WER↓ | EM↑ | CER (ci)↓ | EM (ci)↑ | Ambig↑ | ө/о | ү/у | х | и/й/ь/ъ | ш/щ | Eng↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| byt5-v3-lr5e-4 | 8.95 | 24.76 | 22.56 | 8.15 | 31.79 | 93.64 | 94.74 | 93.85 | 96.89 | 90.09 | 96.39 | 43.59 |

All numbers in %. ci = case-insensitive. Ambig = accuracy on ambiguous letters (ө/о, ү/у, х, и/й/ь/ъ, ш/щ). Eng = Latin words preserved (39 in references).

### valid_v3 (n=2000)

| System | CER↓ | WER↓ | EM↑ | CER (ci)↓ | EM (ci)↑ | Ambig↑ | ө/о | ү/у | х | и/й/ь/ъ | ш/щ | Eng↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| byt5-v3-lr5e-4 | 1.97 | 7.99 | 46.70 | 1.32 | 60.75 | 97.83 | 97.51 | 96.97 | 99.49 | 97.74 | 99.69 | 90.51 |

All numbers in %. ci = case-insensitive. Ambig = accuracy on ambiguous letters (ө/о, ү/у, х, и/й/ь/ъ, ш/щ). Eng = Latin words preserved (790 in references).

### Georgia sample

| Item | Count |
|---|---|
| Records converted | 10,282 |
| Included (status for the five classes is known) | 6,947 |
| Excluded (unmapped codes only, or prolonged QT) | 3,335 |

| Class | Positives | Evaluated? |
|---|---|---|
| NORM | 1,733 | yes |
| MI | 7 | no (fewer than 50 positives) |
| STTC | 3,423 | yes |
| CD | 1,885 | yes |
| HYP | 1,164 | yes |

### Main analysis (all mapped codes)

| Class | 1D ResNet: PTB-XL | 1D ResNet: Georgia (95% CI) | 1D ResNet: change | Classical: PTB-XL | Classical: Georgia (95% CI) | Classical: change |
|---|---|---|---|---|---|---|
| NORM | 0.951 | 0.898 (0.889-0.906) | -0.053 | 0.913 | 0.885 (0.877-0.893) | -0.028 |
| STTC | 0.936 | 0.833 (0.824-0.842) | -0.103 | 0.909 | 0.815 (0.805-0.826) | -0.094 |
| CD | 0.920 | 0.916 (0.909-0.923) | -0.005 | 0.859 | 0.847 (0.837-0.858) | -0.012 |
| HYP | 0.907 | 0.827 (0.814-0.839) | -0.080 | 0.891 | 0.857 (0.846-0.868) | -0.033 |
| Macro average | 0.928 | 0.868 (0.864-0.874) | -0.060 | 0.893 | 0.851 (0.846-0.857) | -0.042 |

### Sensitivity: strict mapping (direct PTB-XL equivalents only)

| Class | 1D ResNet: PTB-XL | 1D ResNet: Georgia (95% CI) | 1D ResNet: change | Classical: PTB-XL | Classical: Georgia (95% CI) | Classical: change |
|---|---|---|---|---|---|---|
| NORM | 0.951 | 0.889 (0.880-0.898) | -0.061 | 0.913 | 0.872 (0.863-0.882) | -0.041 |
| STTC | 0.936 | 0.851 (0.841-0.861) | -0.085 | 0.909 | 0.833 (0.822-0.844) | -0.077 |
| CD | 0.920 | 0.897 (0.888-0.906) | -0.024 | 0.859 | 0.794 (0.778-0.809) | -0.065 |
| HYP | 0.907 | 0.846 (0.832-0.859) | -0.061 | 0.891 | 0.883 (0.871-0.894) | -0.008 |
| Macro average | 0.928 | 0.871 (0.865-0.876) | -0.058 | 0.893 | 0.845 (0.839-0.852) | -0.048 |
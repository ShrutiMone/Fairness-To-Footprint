# Adult Benchmark Summary

- Dataset: `fairlearn.datasets.fetch_adult`
- Rows: 48842
- Sensitive attribute: `sex`
- Constraint: `demographic_parity`

## Holdout Performance

- Baseline: {'Accuracy': 0.8534, 'Precision': 0.7186, 'Recall': 0.6369, 'F1': 0.6753}
- Mitigated: {'Accuracy': 0.8319, 'Precision': 0.7329, 'Recall': 0.4683, 'F1': 0.5715}

## Holdout Fairness (overall)

- Baseline: {'Demographic Parity Difference': 0.1838, 'Equalized Odds Difference': 0.0834, 'False Positive Rate Difference': 0.0834, 'False Negative Rate Difference': 0.0587}
- Mitigated: {'Demographic Parity Difference': 0.0025, 'Equalized Odds Difference': 0.2946, 'False Positive Rate Difference': 0.0443, 'False Negative Rate Difference': 0.2946}

## Discussion starter

Mitigation is successful when fairness disparities (e.g., DP difference / EO difference) drop meaningfully on holdout data while utility metrics (accuracy/F1) remain acceptable for the intended use case.
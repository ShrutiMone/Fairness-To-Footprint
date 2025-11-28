# Using the Mitigated Model

After running mitigation in the web app, you'll receive a **mitigated model file** (`.joblib`). Here's how to use it:

## Installation

Make sure you have the required libraries:
```bash
pip install joblib scikit-learn pandas fairlearn
```

## Loading and Using the Model

### Option 1: Make predictions on new data (simple)

```python
import joblib
import pandas as pd

# Load the model
model_metadata = joblib.load("mitigated_model_<model_id>.joblib")

mitigator = model_metadata["mitigator"]
transformer = model_metadata["transformer"]
label_encoder = model_metadata["label_encoder"]
target_col = model_metadata["target_col"]
sensitive_col = model_metadata["sensitive_col"]
constraint = model_metadata["constraint"]

# Load your new data
df_new = pd.read_csv("new_data.csv")

# Prepare features (same as training: remove target and sensitive columns)
X_new = df_new.drop(columns=[target_col, sensitive_col], errors='ignore')

# Make predictions
predictions = mitigator.predict(X_new)

# Decode predictions back to original labels (if needed)
# For binary classification, predictions are 0 or 1
print("Predictions:", predictions)
```

### Option 2: Evaluate fairness on new data

```python
import joblib
import pandas as pd
from fairlearn.metrics import demographic_parity_difference

# Load the model and new data
model_metadata = joblib.load("mitigated_model_<model_id>.joblib")
df_new = pd.read_csv("new_data.csv")

mitigator = model_metadata["mitigator"]
target_col = model_metadata["target_col"]
sensitive_col = model_metadata["sensitive_col"]

X_new = df_new.drop(columns=[target_col, sensitive_col], errors='ignore')
y_true = df_new[target_col]

# Get predictions
y_pred = mitigator.predict(X_new)

# Compute fairness metric
dpd = demographic_parity_difference(
    y_true=y_true.values,
    y_pred=y_pred,
    sensitive_features=df_new[sensitive_col].values
)

print(f"Demographic Parity Difference: {dpd:.4f}")
print(f"Closer to 0 = fairer")
```

### Option 3: Access model internals

```python
import joblib

model_metadata = joblib.load("mitigated_model_<model_id>.joblib")

mitigator = model_metadata["mitigator"]

# Number of predictors in the mixture
print(f"Number of predictors: {len(mitigator.predictors_)}")

# Mixture weights (how much each predictor contributes)
print(f"Weights: {mitigator.weights_}")

# Constraint used
print(f"Constraint: {model_metadata['constraint']}")

# Training metadata
print(f"Trained at: {model_metadata['timestamp']}")
```

## Key Points

1. **The model is a mixture classifier**: It may use multiple base predictors internally with different weights. The `weights_` array shows their contribution.

2. **Feature order matters**: Your new data must have the same columns (in the same order) as the training data (minus target and sensitive columns).

3. **Sensitive column**: The sensitive column should still be present in your new data for evaluation, but it's NOT used for predictions.

4. **Binary predictions**: The model predicts 0 or 1. For probabilistic outputs, you can access `mitigator.predict_proba()` if the base estimator supports it.

5. **Reproducibility**: The model includes the preprocessing transformer, so predictions automatically handle numeric/categorical encoding.

## Example Workflow

```python
import joblib
import pandas as pd

# 1. Load the model
model_metadata = joblib.load("mitigated_model_abc123.joblib")
mitigator = model_metadata["mitigator"]
target_col = model_metadata["target_col"]
sensitive_col = model_metadata["sensitive_col"]

# 2. Prepare new data
df_test = pd.read_csv("test_data.csv")
X_test = df_test.drop(columns=[target_col, sensitive_col], errors='ignore')

# 3. Make predictions
preds = mitigator.predict(X_test)

# 4. Save results
df_test["prediction"] = preds
df_test.to_csv("predictions_with_mitigated_model.csv", index=False)
```

## Troubleshooting

- **"Column not found" error**: Check that your new data has all the same feature columns as the training data.
- **Shape mismatch**: The number of columns must match. If training had 10 features, new data must have 10 features (same names/order).
- **Import errors**: Ensure you've installed all required packages: `joblib`, `scikit-learn`, `pandas`, `fairlearn`.

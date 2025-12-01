# AI Fairness Audit System

### Backend Setup<br>

cd backend<br>
(optional) python -m venv venv<br>
(optional) venv\Scripts\activate   # on Windows<br>
pip install -r requirements.txt<br>
python app.py<br>

### Frontend Setup<br>
cd frontend<br>
npm install<br>
npm start<br>

## Supported Model Formats

### Machine Learning Models (Full Support)
- **`.joblib`** – scikit-learn models saved with joblib
- **`.pkl`** – scikit-learn models saved with pickle
- These models support both **analysis** and **mitigation** workflows.

### Deep Learning Models (Analysis Only)
- **`.onnx`** – ONNX format (requires `onnxruntime`)
- **`.keras`** or **`.h5`** – TensorFlow/Keras models (requires `tensorflow`)
- **`.pt`** or **`.pth`** – PyTorch models (requires `torch`)

Deep learning models can be analyzed for fairness metrics but **mitigation is not currently supported** because:
1. Re-training DL models requires complete architecture and training code
2. Significant computational overhead (GPU/TPU resources)
3. Fairness constraints in DL are complex and model-specific

### How to Convert Your Model

#### From Scikit-learn (.ipynb notebook)
```python
import joblib
joblib.dump(model, 'my_model.joblib')
```

#### From TensorFlow/Keras to ONNX
```python
import tf2onnx
import onnx
spec = (tf.TensorSpec((None, num_features), tf.float32, name="input"),)
output_path = "model.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
```

#### From PyTorch to ONNX
```python
import torch
import torch.onnx
dummy_input = torch.randn(1, num_features)
torch.onnx.export(model, dummy_input, "model.onnx", input_names=['input'], output_names=['output'])
```

#### Direct PyTorch Save
```python
torch.save(model, 'model.pth')  # Save entire model, not state_dict
```

## API Endpoints

### POST /analyze
- **Purpose**: Compute fairness metrics and baseline analysis
- **Inputs**: CSV data, target column, sensitive attribute, optional model
- **Returns**: Fairness metrics, suggestions, `is_dl_model` flag
- **DL Support**: ✅ Yes (analysis only)

### POST /mitigate
- **Purpose**: Apply fairness mitigation using ExponentiatedGradient
- **Inputs**: CSV data, target column, sensitive attribute, constraint type
- **Returns**: Mitigated model, improved metrics
- **DL Support**: ❌ No (ML models only)

### POST /mitigate_async
- **Purpose**: Asynchronous mitigation for large datasets
- **DL Support**: ❌ No (ML models only)

## Security Notes

- **Model Isolation**: Uploaded models are executed only for prediction; no re-training of user models.
- **Input Validation**: All file uploads and parameters are validated before processing.
- **Computational Limits**: Large datasets (>100k rows) may take longer; use async endpoints for best UX.


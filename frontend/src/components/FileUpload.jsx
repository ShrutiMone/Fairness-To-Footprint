import React, { useState } from "react";

function readCSVHeaders(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      const firstLine = text.split(/\r?\n/)[0];
      const headers = firstLine.split(",").map(h => {
        h = h.trim();
        // Remove surrounding quotes if present
        if ((h.startsWith('"') && h.endsWith('"')) || (h.startsWith("'") && h.endsWith("'"))) {
          h = h.slice(1, -1);
        }
        return h;
      });
      resolve(headers);
    };
    reader.onerror = reject;
    reader.readAsText(file);
  });
}

const FileUpload = ({ onSubmit }) => {
  const [file, setFile] = useState(null);
  const [headers, setHeaders] = useState([]);
  const [target, setTarget] = useState("");
  const [sensitive, setSensitive] = useState("");
  const [predCol, setPredCol] = useState("");
  const [trainBaseline, setTrainBaseline] = useState(true);
  const [modelFile, setModelFile] = useState(null);
  const [wrapModel, setWrapModel] = useState(false);

  const handleFile = async (f) => {
    setFile(f);
    try {
      const h = await readCSVHeaders(f);
      setHeaders(h);
      if (h.length) setTarget(h.includes("Loan_Approved") ? "Loan_Approved" : h[0]);
    } catch (e) {
      console.error(e);
    }
  };

  const submit = (e) => {
    e.preventDefault();
    if (!file || !target || !sensitive) {
      alert("Choose file, target and sensitive columns.");
      return;
    }
    onSubmit(file, target, sensitive, predCol || null, trainBaseline, modelFile, wrapModel);
  };

  return (
    <form id="upload" onSubmit={submit} className="max-w-3xl mx-auto bg-white shadow rounded-lg p-6 mt-8">
      <h2 className="text-xl font-semibold mb-4">Upload Dataset</h2>
      <input type="file" accept=".csv" onChange={(e) => handleFile(e.target.files[0])} className="mb-3"/>
      {headers.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div>
            <label className="block text-sm font-medium">Target column</label>
            <select value={target} onChange={e=>setTarget(e.target.value)} className="mt-1 p-2 border rounded w-full">
              <option value="">Select</option>
              {headers.map(h => <option key={h} value={h}>{h}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium">Sensitive column</label>
            <select value={sensitive} onChange={e=>setSensitive(e.target.value)} className="mt-1 p-2 border rounded w-full">
              <option value="">Select</option>
              {headers.map(h => <option key={h} value={h}>{h}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium">Prediction column (optional)</label>
            <select value={predCol} onChange={e=>setPredCol(e.target.value)} className="mt-1 p-2 border rounded w-full">
              <option value="">None</option>
              {headers.map(h => <option key={h} value={h}>{h}</option>)}
            </select>
          </div>
          <div className="md:col-span-3">
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={trainBaseline} onChange={e=>setTrainBaseline(e.target.checked)} className="mr-2" />
              <span className="text-sm">Train baseline model internally if no predictions provided</span>
            </label>
          </div>
          <div className="md:col-span-3">
            <label className="block text-sm font-medium">Optional pre-trained model (.joblib, .pkl, .onnx, .keras, .pt, .pth)</label>
            <input type="file" accept=".joblib,.pkl,.onnx,.keras,.pt,.pth" onChange={(e)=>setModelFile(e.target.files[0])} className="mt-1" />
            {modelFile && <p className="text-xs text-gray-600 mt-1">Model: {modelFile.name}</p>}
            <label className="flex items-center gap-2 mt-2">
              <input type="checkbox" checked={wrapModel} onChange={e=>setWrapModel(e.target.checked)} className="mr-2" />
              <span className="text-sm">If uploaded model predict fails, apply standard preprocessing and retry</span>
            </label>

            <div className="mt-3 p-3 bg-blue-50 border-l-4 border-blue-200 text-sm rounded">
              <strong>Supported model uploads</strong>
              <ul className="list-disc pl-5 mt-2">
                <li><strong>ML Models (with mitigation support):</strong> Scikit-learn, LightGBM, XGBoost saved as <code>joblib</code> or <code>pickle</code>.</li>
                <li><strong>DL Models (analysis only, no mitigation):</strong> <code>.onnx</code> (ONNX format), <code>.keras</code> (Keras), <code>.pt</code>/<code>.pth</code> (PyTorch). Mitigation button will be disabled for DL models.</li>
              </ul>

              <div className="mt-3">
                <strong>ML Model Conversion:</strong>
                <div className="text-xs text-gray-700 mt-1">
                  <strong>From Jupyter notebook (.ipynb) to joblib/pkl:</strong>
                  <pre className="bg-white p-2 rounded text-xs mt-1">{`# In your notebook after training sklearn model
import joblib
joblib.dump(model, 'model.joblib')

# or with pickle
import pickle
with open('model.pkl','wb') as f:
    pickle.dump(model, f)
`}</pre>
                </div>
              </div>

              <div className="mt-3">
                <strong>DL Model Conversion:</strong>
                <div className="text-xs text-gray-700 mt-1">
                  <strong>TensorFlow/Keras to ONNX:</strong>
                  <pre className="bg-white p-2 rounded text-xs mt-1">{`# Install: pip install tf2onnx
import tf2onnx
import onnx

# Convert saved Keras model to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model)
onnx.save(onnx_model, 'model.onnx')

# Or from SavedModel directory:
# tf2onnx.convert.from_saved_model('./saved_model_dir', output_path='model.onnx')
`}</pre>
                </div>
              </div>

              <div className="mt-3">
                <strong>PyTorch to ONNX:</strong>
                <div className="text-xs text-gray-700 mt-1">
                  <pre className="bg-white p-2 rounded text-xs mt-1">{`import torch
import torch.onnx

# Ensure model is in eval mode
model.eval()

# Dummy input matching your model's expected shape
dummy_input = torch.randn(1, input_size)

# Export to ONNX
torch.onnx.export(model, dummy_input, 'model.onnx', 
                  input_names=['input'], output_names=['output'])
`}</pre>
                </div>
              </div>

              <div className="mt-3">
                <strong>Keras (.h5 or SavedModel):</strong>
                <div className="text-xs text-gray-700 mt-1">
                  Save your model directly in Keras format:
                  <pre className="bg-white p-2 rounded text-xs mt-1">{`# Save as .keras (recommended)
model.save('model.keras')

# or legacy .h5 format
model.save('model.h5')
`}</pre>
                </div>
              </div>

              <div className="mt-3">
                <strong>PyTorch (.pt / .pth):</strong>
                <div className="text-xs text-gray-700 mt-1">
                  Save PyTorch models directly:
                  <pre className="bg-white p-2 rounded text-xs mt-1">{`import torch

# Save entire model
torch.save(model, 'model.pth')

# or save state dict (recommended for reproducibility)
torch.save(model.state_dict(), 'model_weights.pth')
`}</pre>
                </div>
              </div>

              <div className="mt-3 text-red-600 text-xs">
                <strong>Security note:</strong> Loading pickle/joblib files can execute code. Only upload model files you trust. ONNX, Keras, and PyTorch formats are safer.
              </div>
            </div>
          </div>
        </div>
      )}
      <button type="submit" className="mt-4 bg-blue-600 text-white px-4 py-2 rounded">Analyze Fairness</button>
    </form>
  );
};

export default FileUpload;

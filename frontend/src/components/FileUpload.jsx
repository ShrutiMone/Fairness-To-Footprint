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
            <label className="block text-sm font-medium">Optional pre-trained model (.joblib or .pkl)</label>
            <input type="file" accept=".joblib,.pkl" onChange={(e)=>setModelFile(e.target.files[0])} className="mt-1" />
            {modelFile && <p className="text-xs text-gray-600 mt-1">Model: {modelFile.name}</p>}
            <label className="flex items-center gap-2 mt-2">
              <input type="checkbox" checked={wrapModel} onChange={e=>setWrapModel(e.target.checked)} className="mr-2" />
              <span className="text-sm">If uploaded model predict fails, apply standard preprocessing and retry</span>
            </label>
          </div>
        </div>
      )}
      <button type="submit" className="mt-4 bg-blue-600 text-white px-4 py-2 rounded">Analyze Fairness</button>
    </form>
  );
};

export default FileUpload;

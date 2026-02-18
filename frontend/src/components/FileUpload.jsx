// import React, { useState } from "react";

// function readCSVHeaders(file) {
//   return new Promise((resolve, reject) => {
//     const reader = new FileReader();
//     reader.onload = (e) => {
//       const text = e.target.result;
//       const firstLine = text.split(/\r?\n/)[0];
//       const headers = firstLine.split(",").map(h => {
//         h = h.trim();
//         // Remove surrounding quotes if present
//         if ((h.startsWith('"') && h.endsWith('"')) || (h.startsWith("'") && h.endsWith("'"))) {
//           h = h.slice(1, -1);
//         }
//         return h;
//       });
//       resolve(headers);
//     };
//     reader.onerror = reject;
//     reader.readAsText(file);
//   });
// }

// const FileUpload = ({ onSubmit }) => {
//   const [file, setFile] = useState(null);
//   const [headers, setHeaders] = useState([]);
//   const [target, setTarget] = useState("");
//   const [sensitive, setSensitive] = useState("");
//   const [predCol, setPredCol] = useState("");
//   const [trainBaseline, setTrainBaseline] = useState(true);
//   const [modelFile, setModelFile] = useState(null);
//   const [wrapModel, setWrapModel] = useState(false);

//   const handleFile = async (f) => {
//     setFile(f);
//     try {
//       const h = await readCSVHeaders(f);
//       setHeaders(h);
//       if (h.length) setTarget(h.includes("Loan_Approved") ? "Loan_Approved" : h[0]);
//     } catch (e) {
//       console.error(e);
//     }
//   };

//   const submit = (e) => {
//     e.preventDefault();
//     if (!file || !target || !sensitive) {
//       alert("Choose file, target and sensitive columns.");
//       return;
//     }
//     onSubmit(file, target, sensitive, predCol || null, trainBaseline, modelFile, wrapModel);
//   };

//   return (
//     <form id="upload" onSubmit={submit} className="max-w-3xl mx-auto bg-white shadow rounded-lg p-6 mt-8">
//       <h2 className="text-xl font-semibold mb-4">Upload Dataset</h2>
//       <input type="file" accept=".csv" onChange={(e) => handleFile(e.target.files[0])} className="mb-3"/>
//       {headers.length > 0 && (
//         <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
//           <div>
//             <label className="block text-sm font-medium">Target column</label>
//             <select value={target} onChange={e=>setTarget(e.target.value)} className="mt-1 p-2 border rounded w-full">
//               <option value="">Select</option>
//               {headers.map(h => <option key={h} value={h}>{h}</option>)}
//             </select>
//           </div>
//           <div>
//             <label className="block text-sm font-medium">Sensitive column</label>
//             <select value={sensitive} onChange={e=>setSensitive(e.target.value)} className="mt-1 p-2 border rounded w-full">
//               <option value="">Select</option>
//               {headers.map(h => <option key={h} value={h}>{h}</option>)}
//             </select>
//           </div>
//           <div>
//             <label className="block text-sm font-medium">Prediction column (optional)</label>
//             <select value={predCol} onChange={e=>setPredCol(e.target.value)} className="mt-1 p-2 border rounded w-full">
//               <option value="">None</option>
//               {headers.map(h => <option key={h} value={h}>{h}</option>)}
//             </select>
//           </div>
//           <div className="md:col-span-3">
//             <label className="flex items-center gap-2">
//               <input type="checkbox" checked={trainBaseline} onChange={e=>setTrainBaseline(e.target.checked)} className="mr-2" />
//               <span className="text-sm">Train baseline model internally if no predictions provided</span>
//             </label>
//           </div>
//           <div className="md:col-span-3">
//             <label className="block text-sm font-medium">Optional pre-trained model (.joblib, .pkl, .onnx, .keras, .pt, .pth)</label>
//             <input type="file" accept=".joblib,.pkl,.onnx,.keras,.pt,.pth" onChange={(e)=>setModelFile(e.target.files[0])} className="mt-1" />
//             {modelFile && <p className="text-xs text-gray-600 mt-1">Model: {modelFile.name}</p>}
//             <label className="flex items-center gap-2 mt-2">
//               <input type="checkbox" checked={wrapModel} onChange={e=>setWrapModel(e.target.checked)} className="mr-2" />
//               <span className="text-sm">If uploaded model predict fails, apply standard preprocessing and retry</span>
//             </label>

//             <div className="mt-3 p-3 bg-blue-50 border-l-4 border-blue-200 text-sm rounded">
//               <strong>Supported model uploads</strong>
//               <ul className="list-disc pl-5 mt-2">
//                 <li><strong>ML Models (with mitigation support):</strong> Scikit-learn, LightGBM, XGBoost saved as <code>joblib</code> or <code>pickle</code>.</li>
//                 <li><strong>DL Models (analysis only, no mitigation):</strong> <code>.onnx</code> (ONNX format), <code>.keras</code> (Keras), <code>.pt</code>/<code>.pth</code> (PyTorch). Mitigation button will be disabled for DL models.</li>
//               </ul>

//               <div className="mt-3">
//                 <strong>ML Model Conversion:</strong>
//                 <div className="text-xs text-gray-700 mt-1">
//                   <strong>From Jupyter notebook (.ipynb) to joblib/pkl:</strong>
//                   <pre className="bg-white p-2 rounded text-xs mt-1">{`# In your notebook after training sklearn model
// import joblib
// joblib.dump(model, 'model.joblib')

// # or with pickle
// import pickle
// with open('model.pkl','wb') as f:
//     pickle.dump(model, f)
// `}</pre>
//                 </div>
//               </div>

//               <div className="mt-3">
//                 <strong>DL Model Conversion:</strong>
//                 <div className="text-xs text-gray-700 mt-1">
//                   <strong>TensorFlow/Keras to ONNX:</strong>
//                   <pre className="bg-white p-2 rounded text-xs mt-1">{`# Install: pip install tf2onnx
// import tf2onnx
// import onnx

// # Convert saved Keras model to ONNX
// onnx_model, _ = tf2onnx.convert.from_keras(model)
// onnx.save(onnx_model, 'model.onnx')

// # Or from SavedModel directory:
// # tf2onnx.convert.from_saved_model('./saved_model_dir', output_path='model.onnx')
// `}</pre>
//                 </div>
//               </div>

//               <div className="mt-3">
//                 <strong>PyTorch to ONNX:</strong>
//                 <div className="text-xs text-gray-700 mt-1">
//                   <pre className="bg-white p-2 rounded text-xs mt-1">{`import torch
// import torch.onnx

// # Ensure model is in eval mode
// model.eval()

// # Dummy input matching your model's expected shape
// dummy_input = torch.randn(1, input_size)

// # Export to ONNX
// torch.onnx.export(model, dummy_input, 'model.onnx', 
//                   input_names=['input'], output_names=['output'])
// `}</pre>
//                 </div>
//               </div>

//               <div className="mt-3">
//                 <strong>Keras (.h5 or SavedModel):</strong>
//                 <div className="text-xs text-gray-700 mt-1">
//                   Save your model directly in Keras format:
//                   <pre className="bg-white p-2 rounded text-xs mt-1">{`# Save as .keras (recommended)
// model.save('model.keras')

// # or legacy .h5 format
// model.save('model.h5')
// `}</pre>
//                 </div>
//               </div>

//               <div className="mt-3">
//                 <strong>PyTorch (.pt / .pth):</strong>
//                 <div className="text-xs text-gray-700 mt-1">
//                   Save PyTorch models directly:
//                   <pre className="bg-white p-2 rounded text-xs mt-1">{`import torch

// # Save entire model
// torch.save(model, 'model.pth')

// # or save state dict (recommended for reproducibility)
// torch.save(model.state_dict(), 'model_weights.pth')
// `}</pre>
//                 </div>
//               </div>

//               <div className="mt-3 text-red-600 text-xs">
//                 <strong>Security note:</strong> Loading pickle/joblib files can execute code. Only upload model files you trust. ONNX, Keras, and PyTorch formats are safer.
//               </div>
//             </div>
//           </div>
//         </div>
//       )}
//       <button type="submit" className="mt-4 bg-blue-600 text-white px-4 py-2 rounded">Analyze Fairness</button>
//     </form>
//   );
// };

// export default FileUpload;

import React, { useState, useRef } from "react";
import { T } from "../theme";

/* ── helpers ──────────────────────────────────────────────────────────────── */
function readCSVHeaders(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const firstLine = e.target.result.split(/\r?\n/)[0];
      const headers = firstLine.split(",").map(h => {
        h = h.trim();
        if ((h.startsWith('"') && h.endsWith('"')) || (h.startsWith("'") && h.endsWith("'")))
          h = h.slice(1, -1);
        return h;
      });
      resolve(headers);
    };
    reader.onerror = reject;
    reader.readAsText(file);
  });
}

/* ── tiny subcomponents ───────────────────────────────────────────────────── */
function Label({ children }) {
  return (
    <div style={{
      color: T.textDim, fontSize: 11, fontWeight: 700,
      textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 6,
    }}>
      {children}
    </div>
  );
}

function Select({ value, onChange, options, placeholder }) {
  return (
    <select
      value={value}
      onChange={onChange}
      style={{
        width: "100%", padding: "8px 10px",
        background: T.surfaceHi, border: `1px solid ${T.border}`,
        borderRadius: 6, color: value ? T.text : T.textDim,
        fontSize: 13, fontFamily: T.font, cursor: "pointer", outline: "none",
      }}
    >
      <option value="">{placeholder || "Select…"}</option>
      {options.map(h => <option key={h} value={h}>{h}</option>)}
    </select>
  );
}

function Toggle({ checked, onChange, label, sub }) {
  return (
    <label style={{ display: "flex", alignItems: "flex-start", gap: 10, cursor: "pointer" }}>
      <div
        onClick={() => onChange(!checked)}
        style={{
          width: 36, height: 20, borderRadius: 10, flexShrink: 0, marginTop: 1,
          background: checked ? T.amber : T.border,
          position: "relative", transition: "background .2s", cursor: "pointer",
        }}
      >
        <div style={{
          position: "absolute", top: 2, left: checked ? 18 : 2,
          width: 16, height: 16, borderRadius: "50%", background: "#fff",
          transition: "left .2s", boxShadow: "0 1px 3px #0005",
        }} />
      </div>
      <div>
        <div style={{ color: T.text, fontSize: 13, fontWeight: 600 }}>{label}</div>
        {sub && <div style={{ color: T.textDim, fontSize: 11, marginTop: 2 }}>{sub}</div>}
      </div>
    </label>
  );
}

function CodeSnippet({ code }) {
  return (
    <pre style={{
      background: "#0a0c10", border: `1px solid ${T.border}`,
      borderRadius: 6, padding: "10px 12px", margin: "6px 0 0",
      fontSize: 11, color: T.sky, overflowX: "auto", lineHeight: 1.6,
      fontFamily: "'SF Mono', 'Fira Code', monospace",
    }}>
      {code.trim()}
    </pre>
  );
}

function AccordionItem({ title, children }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ borderBottom: `1px solid ${T.border}` }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: "100%", display: "flex", justifyContent: "space-between",
          alignItems: "center", padding: "10px 14px",
          background: "transparent", border: "none", cursor: "pointer",
          color: T.text, fontSize: 13, fontWeight: 600, fontFamily: T.font,
          textAlign: "left",
        }}
      >
        <span>{title}</span>
        <span style={{ color: T.textDim, fontSize: 14 }}>{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <div style={{ padding: "0 14px 12px" }}>
          {children}
        </div>
      )}
    </div>
  );
}

/* ── drop zone ────────────────────────────────────────────────────────────── */
function DropZone({ file, onFile }) {
  const ref = useRef();
  const [drag, setDrag] = useState(false);

  const handle = async (f) => {
    if (!f || !f.name.endsWith(".csv")) return alert("Please select a .csv file.");
    onFile(f);
  };

  return (
    <div
      onDrop={e => { e.preventDefault(); setDrag(false); handle(e.dataTransfer.files[0]); }}
      onDragOver={e => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onClick={() => ref.current.click()}
      style={{
        border: `2px dashed ${drag ? T.amber : file ? T.green : T.border}`,
        borderRadius: 10, padding: "32px 24px", textAlign: "center",
        cursor: "pointer", background: drag ? T.amberDim : file ? T.greenDim : T.surface,
        transition: "all .2s", userSelect: "none",
      }}
    >
      <input ref={ref} type="file" accept=".csv" style={{ display: "none" }}
        onChange={e => handle(e.target.files[0])} />
      <div style={{ fontSize: 28, marginBottom: 8 }}>{file ? "✅" : "📂"}</div>
      {file ? (
        <>
          <div style={{ color: T.green, fontWeight: 700, fontSize: 14 }}>{file.name}</div>
          <div style={{ color: T.textDim, fontSize: 12, marginTop: 4 }}>Click to replace</div>
        </>
      ) : (
        <>
          <div style={{ color: T.text, fontWeight: 600, fontSize: 14 }}>
            Drop your <span style={{ color: T.amber }}>dataset CSV</span> here
          </div>
          <div style={{ color: T.textDim, fontSize: 12, marginTop: 6 }}>
            or click to browse
          </div>
        </>
      )}
    </div>
  );
}

/* ── model format guide (collapsible) ────────────────────────────────────── */
function ModelFormatGuide() {
  const [open, setOpen] = useState(false);
  return (
    <div style={{
      border: `1px solid ${T.border}`, borderRadius: 8,
      background: T.surface, overflow: "hidden",
    }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: "100%", display: "flex", justifyContent: "space-between",
          alignItems: "center", padding: "11px 14px",
          background: "transparent", border: "none", cursor: "pointer",
          color: T.sky, fontSize: 13, fontWeight: 700, fontFamily: T.font,
        }}
      >
        <span>📖 Model Format & Conversion Guide</span>
        <span style={{ color: T.textDim }}>{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div style={{ borderTop: `1px solid ${T.border}` }}>
          {/* Support matrix */}
          <div style={{ padding: "12px 14px 0" }}>
            <div style={{ color: T.textDim, fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
              Supported Formats
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
              {[
                { fmt: ".joblib / .pkl", label: "Scikit-learn", badge: "Full support", color: T.green },
                { fmt: ".onnx", label: "ONNX", badge: "Analysis only", color: T.amber },
                { fmt: ".keras / .h5", label: "Keras / TF", badge: "Analysis only", color: T.amber },
                { fmt: ".pt / .pth", label: "PyTorch", badge: "Analysis only", color: T.amber },
              ].map(({ fmt, label, badge, color }) => (
                <div key={fmt} style={{
                  background: T.surfaceHi, border: `1px solid ${T.border}`,
                  borderRadius: 6, padding: "8px 10px",
                  display: "flex", alignItems: "center", justifyContent: "space-between",
                }}>
                  <div>
                    <code style={{ color: T.violet, fontSize: 12 }}>{fmt}</code>
                    <div style={{ color: T.textDim, fontSize: 11, marginTop: 2 }}>{label}</div>
                  </div>
                  <span style={{
                    fontSize: 10, fontWeight: 700, padding: "2px 7px", borderRadius: 10,
                    background: color + "22", color, border: `1px solid ${color}44`,
                  }}>{badge}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Conversion snippets */}
          <div style={{ padding: "12px 14px" }}>
            <div style={{ color: T.textDim, fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
              Conversion Snippets
            </div>
            <div style={{ border: `1px solid ${T.border}`, borderRadius: 6, overflow: "hidden" }}>
              <AccordionItem title="Scikit-learn → joblib (recommended)">
                <CodeSnippet code={`import joblib\njoblib.dump(model, 'model.joblib')`} />
              </AccordionItem>
              <AccordionItem title="TensorFlow / Keras → ONNX">
                <CodeSnippet code={`import tf2onnx, onnx\nonnx_model, _ = tf2onnx.convert.from_keras(model)\nonnx.save(onnx_model, 'model.onnx')`} />
              </AccordionItem>
              <AccordionItem title="PyTorch → ONNX">
                <CodeSnippet code={`import torch\nmodel.eval()\ndummy = torch.randn(1, input_size)\ntorch.onnx.export(model, dummy, 'model.onnx',\n    input_names=['input'], output_names=['output'])`} />
              </AccordionItem>
              <AccordionItem title="Keras → .keras / .h5">
                <CodeSnippet code={`model.save('model.keras')   # recommended\nmodel.save('model.h5')      # legacy`} />
              </AccordionItem>
              <AccordionItem title="PyTorch → .pth (full model)">
                <CodeSnippet code={`torch.save(model, 'model.pth')  # full model, not state_dict`} />
              </AccordionItem>
            </div>
            <div style={{
              marginTop: 10, padding: "8px 10px", borderRadius: 6,
              background: T.redDim, border: `1px solid ${T.red}44`,
              color: T.red, fontSize: 11,
            }}>
              ⚠ Security: only upload model files you trust. Pickle/joblib files can execute code.
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ── main component ───────────────────────────────────────────────────────── */
const FileUpload = ({ onSubmit }) => {
  const [file, setFile]                 = useState(null);
  const [headers, setHeaders]           = useState([]);
  const [target, setTarget]             = useState("");
  const [sensitive, setSensitive]       = useState("");
  const [predCol, setPredCol]           = useState("");
  const [trainBaseline, setTrainBaseline] = useState(true);
  const [modelFile, setModelFile]       = useState(null);
  const [wrapModel, setWrapModel]       = useState(false);
  const modelRef                        = useRef();

  const handleFile = async (f) => {
    setFile(f);
    try {
      const h = await readCSVHeaders(f);
      setHeaders(h);
      if (h.length) setTarget(h[0]);
    } catch (e) { console.error(e); }
  };

  const submit = (e) => {
    e.preventDefault();
    if (!file || !target || !sensitive) {
      alert("Please select a CSV file, and choose target + sensitive columns.");
      return;
    }
    onSubmit(file, target, sensitive, predCol || null, trainBaseline, modelFile, wrapModel);
  };

  const Card = ({ children, style = {} }) => (
    <div style={{
      background: T.surface, border: `1px solid ${T.border}`,
      borderRadius: 10, padding: "18px 20px", ...style,
    }}>
      {children}
    </div>
  );

  const SectionTitle = ({ children, accent }) => (
    <div style={{
      color: "#fff", fontSize: 13, fontWeight: 700, marginBottom: 14,
      paddingBottom: 10, borderBottom: `1px solid ${T.border}`,
      display: "flex", alignItems: "center", gap: 8,
    }}>
      {accent && <span style={{ color: accent }}>{accent === T.amber ? "◆" : "◆"}</span>}
      {children}
    </div>
  );

  return (
    <div id="upload" style={{ maxWidth: 780, margin: "0 auto", padding: "32px 0", fontFamily: T.font }}>

      {/* Page header */}
      <div style={{ textAlign: "center", marginBottom: 32 }}>
        <div style={{ fontSize: 36, marginBottom: 8 }}>⚖️</div>
        <h1 style={{
          color: "#fff", fontSize: 24, fontWeight: 800,
          margin: "0 0 6px", letterSpacing: "-0.02em",
        }}>
          AI Fairness <span style={{ color: T.amber }}>Audit</span>
        </h1>
        <p style={{ color: T.textDim, fontSize: 13, margin: 0 }}>
          Upload your dataset and configure the audit to analyse model fairness across sensitive groups.
        </p>
      </div>

      <form onSubmit={submit} style={{ display: "flex", flexDirection: "column", gap: 16 }}>

        {/* Step 1 — Dataset */}
        <Card style={{ borderTop: `3px solid ${T.amber}` }}>
          <SectionTitle>Step 1 — Upload Dataset</SectionTitle>
          <DropZone file={file} onFile={handleFile} />
        </Card>

        {/* Step 2 — Columns (only show after CSV loaded) */}
        {headers.length > 0 && (
          <Card style={{ borderTop: `3px solid ${T.sky}` }}>
            <SectionTitle>Step 2 — Configure Columns</SectionTitle>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 14 }}>
              <div>
                <Label>Target column *</Label>
                <Select value={target} onChange={e => setTarget(e.target.value)}
                  options={headers} placeholder="Select target" />
              </div>
              <div>
                <Label>Sensitive attribute *</Label>
                <Select value={sensitive} onChange={e => setSensitive(e.target.value)}
                  options={headers} placeholder="Select sensitive" />
              </div>
              <div>
                <Label>Prediction column <span style={{ color: T.textDim, fontWeight: 400 }}>(optional)</span></Label>
                <Select value={predCol} onChange={e => setPredCol(e.target.value)}
                  options={headers} placeholder="None" />
              </div>
            </div>
          </Card>
        )}

        {/* Step 3 — Model options */}
        {headers.length > 0 && (
          <Card style={{ borderTop: `3px solid ${T.violet}` }}>
            <SectionTitle>Step 3 — Model Options</SectionTitle>
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>

              <Toggle
                checked={trainBaseline}
                onChange={setTrainBaseline}
                label="Train baseline model internally"
                sub="If no model or prediction column is provided, a LogisticRegression baseline is trained automatically."
              />

              {/* Model file upload */}
              <div>
                <Label>Upload pre-trained model <span style={{ color: T.textDim, fontWeight: 400 }}>(optional)</span></Label>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <button
                    type="button"
                    onClick={() => modelRef.current.click()}
                    style={{
                      padding: "7px 16px", borderRadius: 6,
                      border: `1px solid ${modelFile ? T.green : T.border}`,
                      background: modelFile ? T.greenDim : T.surfaceHi,
                      color: modelFile ? T.green : T.text,
                      fontSize: 12, fontWeight: 600, cursor: "pointer", fontFamily: T.font,
                    }}
                  >
                    {modelFile ? `✓ ${modelFile.name}` : "Browse model file…"}
                  </button>
                  {modelFile && (
                    <button
                      type="button"
                      onClick={() => setModelFile(null)}
                      style={{
                        background: "none", border: "none",
                        color: T.textDim, cursor: "pointer", fontSize: 18,
                      }}
                    >×</button>
                  )}
                  <input
                    ref={modelRef}
                    type="file"
                    accept=".joblib,.pkl,.onnx,.keras,.h5,.pt,.pth"
                    style={{ display: "none" }}
                    onChange={e => setModelFile(e.target.files[0] || null)}
                  />
                </div>
                <div style={{ color: T.textDim, fontSize: 11, marginTop: 5 }}>
                  Accepts: <code>.joblib</code> <code>.pkl</code> <code>.onnx</code> <code>.keras</code> <code>.pt</code> <code>.pth</code>
                </div>
              </div>

              {modelFile && (
                <Toggle
                  checked={wrapModel}
                  onChange={setWrapModel}
                  label="Apply standard preprocessing if prediction fails"
                  sub="Fits a ColumnTransformer on your data and retries inference. Enable if you're seeing prediction errors."
                />
              )}

              {/* Collapsible format guide */}
              <ModelFormatGuide />
            </div>
          </Card>
        )}

        {/* Submit */}
        {headers.length > 0 && (
          <button
            type="submit"
            style={{
              padding: "12px 28px", borderRadius: 8,
              background: `linear-gradient(135deg, ${T.amber}, #e07b00)`,
              border: "none", color: "#000", fontSize: 14,
              fontWeight: 800, cursor: "pointer", fontFamily: T.font,
              letterSpacing: "0.01em", transition: "opacity .15s",
              alignSelf: "flex-start",
            }}
            onMouseEnter={e => e.target.style.opacity = "0.85"}
            onMouseLeave={e => e.target.style.opacity = "1"}
          >
            Analyze Fairness →
          </button>
        )}
      </form>
    </div>
  );
};

export default FileUpload;
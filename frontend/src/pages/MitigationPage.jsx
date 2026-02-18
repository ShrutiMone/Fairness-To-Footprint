// import React, { useState, useRef, useEffect } from "react";
// import { mitigateDataset, mitigateUserModel, mitigateDatasetAsync, mitigateUserModelAsync, getProgress, getResult } from "../utils/api";

// const MitigationPage = ({ uploadedFile, selectedTarget, selectedSensitive, uploadedModel, isDLModel }) => {
//   const [file, setFile] = useState(uploadedFile || null);
//   const [userModel, setUserModel] = useState(uploadedModel || null);
//   const [target, setTarget] = useState(selectedTarget || "");
//   const [sensitive, setSensitive] = useState(selectedSensitive || "");
//   const [mode, setMode] = useState(uploadedModel ? "user_model" : "builtin"); // "builtin" or "user_model"
//   useEffect(() => {
//     if (uploadedFile) setFile(uploadedFile);
//     if (selectedTarget) setTarget(selectedTarget);
//     if (selectedSensitive) setSensitive(selectedSensitive);
//     if (uploadedModel) {
//       setUserModel(uploadedModel);
//       setMode("user_model");
//     }
//   }, [uploadedFile, selectedTarget, selectedSensitive, uploadedModel]);
//   const [constraint, setConstraint] = useState("demographic_parity");
//   const [perfPreference, setPerfPreference] = useState("precision");
//   const [result, setResult] = useState(null);
//   const [loading, setLoading] = useState(false);
//   const [jobPercent, setJobPercent] = useState(0);
//   const [jobMessage, setJobMessage] = useState("");
//   const pollRef = useRef(null);

//   const run = async () => {
//     if (!file || !target || !sensitive) { alert("Choose file, target and sensitive"); return; }
    
//     setLoading(true);
//     try {
//       // Start async mitigation job and poll progress
//       let startRes;
//       if (mode === "builtin") {
//         startRes = await mitigateDatasetAsync(file, target, sensitive, constraint);
//       } else {
//         if (!userModel) { alert("Upload your model file"); setLoading(false); return; }
//         // we reuse the same async endpoint for user models for simplicity
//         startRes = await mitigateUserModelAsync(file, userModel, target, sensitive, constraint);
//       }

//       const jobId = startRes.job_id;
//       if (!jobId) {
//         // fallback: maybe returned immediate result
//         setResult(startRes);
//         setLoading(false);
//         return;
//       }

//       setJobPercent(0);
//       setJobMessage("queued");

//       // poll progress
//       pollRef.current = setInterval(async () => {
//         try {
//           const p = await getProgress(jobId);
//           setJobPercent(p.percent || 0);
//           setJobMessage(p.message || "running");
//           if (p.status === "done") {
//             clearInterval(pollRef.current);
//             const final = await getResult(jobId);
//             setResult(final);
//             setLoading(false);
//           } else if (p.status === "failed") {
//             clearInterval(pollRef.current);
//             const final = await getResult(jobId);
//             setResult(final);
//             setLoading(false);
//           }
//         } catch (err) {
//           console.error(err);
//           clearInterval(pollRef.current);
//           setLoading(false);
//         }
//       }, 1000);
//     } catch (err) {
//       console.error(err);
//       setResult({ error: String(err) });
//       setLoading(false);
//     }
//   };

//   // Derived view data for nicer rendering
//   const overallBaseline = result?.metrics_baseline?.overall || {};
//   const overall = result?.metrics_after_mitigation?.overall || {};
//   const overallBaselineTest = result?.metrics_baseline_test?.overall || {};
//   const overallTest = result?.metrics_after_mitigation_test?.overall || {};
//   const perfBaseline = result?.performance_baseline || {};
//   const perfAfter = result?.performance_after_mitigation || {};
//   const perfBaselineTest = result?.performance_baseline_test || {};
//   const perfAfterTest = result?.performance_after_mitigation_test || {};
//   const byGroupBaseline = result?.metrics_baseline?.by_group || {};
//   const byGroup = result?.metrics_after_mitigation?.by_group || {};
//   const suggestions = result?.suggestions || [];
//   const predictions = result?.predictions || [];
//   const posCount = predictions.filter((p) => p === 1).length;
//   const negCount = predictions.length - posCount;
//   const posPct = predictions.length ? ((posCount / predictions.length) * 100).toFixed(1) : "0.0";
//   const fmt = (v) => (typeof v === "number" ? v : v);
//   const improvement = (baseline, after) => {
//     if (typeof baseline === "number" && typeof after === "number") {
//       const delta = baseline - after;
//       const color = delta > 0 ? "text-green-600" : delta < 0 ? "text-red-600" : "text-gray-600";
//       const arrow = delta > 0 ? "↓" : delta < 0 ? "↑" : "→";
//       return <span className={color}> {arrow} {Math.abs(delta).toFixed(4)}</span>;
//     }
//     return null;
//   };
//   return (
//     <div id="mitigation" className="max-w-4xl mx-auto mt-8 bg-white p-6 rounded-lg shadow">
//       <h3 className="text-xl font-semibold mb-4">Fairness Mitigation</h3>
      
//       <div className="mb-4 border-b pb-4 space-y-3">
//         <div>
//           <label className="block text-sm font-semibold mb-2">Mitigation Mode:</label>
//           <div className="flex gap-4">
//             <label className="flex items-center">
//               <input type="radio" value="builtin" checked={mode === "builtin"} onChange={(e) => { setMode(e.target.value); }} className="mr-2" />
//               Use baseline model from analysis
//             </label>
//             <label className="flex items-center">
//               <input type="radio" value="user_model" checked={mode === "user_model"} onChange={(e) => { setMode(e.target.value); }} className="mr-2" />
//               Use your uploaded model
//             </label>
//           </div>
//         </div>

//         <div>
//           <label className="block text-sm font-semibold mb-2">Fairness Constraint:</label>
//           <select value={constraint} onChange={e=>setConstraint(e.target.value)} className="border p-2 rounded">
//             <option value="demographic_parity">Demographic Parity (equal selection rates)</option>
//             <option value="equalized_odds">Equalized Odds (equal error rates)</option>
//           </select>
//         </div>
//         <div>
//           <label className="block text-sm font-semibold mb-2">Optimize for:</label>
//           <div className="flex gap-4 text-sm">
//             <label className="flex items-center">
//               <input type="radio" name="perfPref" value="precision" checked={perfPreference === "precision"} onChange={(e)=>setPerfPreference(e.target.value)} className="mr-2" />
//               Precision
//             </label>
//             <label className="flex items-center">
//               <input type="radio" name="perfPref" value="recall" checked={perfPreference === "recall"} onChange={(e)=>setPerfPreference(e.target.value)} className="mr-2" />
//               Recall
//             </label>
//           </div>
//           <div className="text-xs text-gray-600 mt-1">
//             This only affects how you interpret results today. It does not change the mitigation algorithm yet.
//           </div>
//         </div>
//       </div>
      
//       <div className="space-y-3 mb-4">
//         {/* If the page was reached from Analyze, we already have file/target/sensitive; show a compact view */}
//         {file && target && sensitive ? (
//           <div className="text-sm text-gray-700 bg-gray-50 p-3 rounded">
//             <div>📄 Data: <span className="font-medium">{file.name}</span></div>
//             <div>🎯 Target: <span className="font-medium">{target}</span></div>
//             <div>👤 Sensitive: <span className="font-medium">{sensitive}</span></div>
//             {mode === "user_model" && userModel && (
//               <div>🤖 Model: <span className="font-medium">{userModel.name}</span></div>
//             )}
//           </div>
//         ) : (
//           <>
//             <div>
//               <label className="block text-sm font-semibold mb-1">Data File (CSV)</label>
//               <input type="file" accept=".csv" onChange={(e)=>setFile(e.target.files[0])} className="border p-2 w-full rounded" />
//               {file && <p className="text-xs text-gray-600 mt-1">File: {file.name}</p>}
//             </div>

//             <div>
//               <label className="block text-sm font-semibold mb-1">Target Column</label>
//               <input placeholder="e.g., 'approved'" value={target} onChange={e=>setTarget(e.target.value)} className="border p-2 w-full rounded" />
//             </div>

//             <div>
//               <label className="block text-sm font-semibold mb-1">Sensitive Attribute Column</label>
//               <input placeholder="e.g., 'gender'" value={sensitive} onChange={e=>setSensitive(e.target.value)} className="border p-2 w-full rounded" />
//             </div>
//           </>
//         )}
//       </div>

//       <button onClick={run} disabled={loading || isDLModel} className="bg-green-600 text-white px-6 py-2 rounded font-semibold hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed">
//         {isDLModel ? "Mitigation not supported for deep-learning models" : loading ? "Running mitigation..." : "Run Mitigation"}
//       </button>

//       {isDLModel && (
//         <div className="mt-3 p-3 bg-yellow-100 border border-yellow-400 text-yellow-800 rounded">
//           <p className="text-sm">
//             <strong>Note:</strong> Mitigation is not currently supported for deep-learning models (ONNX, Keras, PyTorch). 
//             You can view the fairness analysis metrics above, but automated mitigation is available for scikit-learn and similar ML models only.
//           </p>
//         </div>
//       )}

//       {loading && (
//         <div className="mt-4">
//           <div className="text-sm text-gray-600 mb-2">{jobMessage} — {jobPercent}%</div>
//           <div className="w-full bg-gray-200 rounded-full h-3">
//             <div className="bg-blue-600 h-3 rounded-full" style={{ width: `${jobPercent}%` }} />
//           </div>
//         </div>
//       )}

//       {result && (
//         <div className="mt-6">
//           <h4 className="font-semibold mb-3">Mitigation Result</h4>
//           {result.error ? (
//             <div className="text-red-600">{result.error}</div>
//           ) : (
//             <>
//               {/* Suggestions Section */}
//               {suggestions && suggestions.length > 0 && (
//                 <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 rounded-lg mb-4 shadow-sm">
//                   <h5 className="text-lg font-semibold mb-2 text-yellow-800">Suggested Improvements for Dataset</h5>
//                   <ul className="list-disc pl-6 text-yellow-900 space-y-1 text-sm">
//                     {suggestions.map((s, i) => (
//                       <li key={i}>{s}</li>
//                     ))}
//                   </ul>
//                 </div>
//               )}

//               {result.model_download_url && (
//                 <div className="mb-4">
//                   <a
//                     href={result.model_download_url}
//                     download
//                     className="bg-blue-600 text-white px-4 py-2 rounded inline-block hover:bg-blue-700"
//                   >
//                     📥 Download Mitigated Model
//                   </a>
//                   <p className="text-xs text-gray-600 mt-1">
//                     Model ID: {result.model_id}
//                   </p>
//                 </div>
//               )}

//               {/* Strategy / estimate */}
//               {(result.strategy || result.time_estimate_seconds) && (
//                 <div className="mb-4 text-sm text-gray-700">
//                   {result.strategy && <div>Strategy used: <span className="font-medium">{result.strategy}</span></div>}
//                   {result.time_estimate_seconds && <div>Estimated time: <span className="font-medium">{result.time_estimate_seconds}s</span></div>}
//                 </div>
//               )}

//               <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
//                 <div className="bg-white border rounded p-4 shadow-sm">
//                   <h5 className="font-semibold mb-2">Before Mitigation</h5>
//                   <ul className="text-sm space-y-1">
//                     {Object.entries(overallBaseline).length === 0 && <li className="text-gray-500">No baseline metrics</li>}
//                     {Object.entries(overallBaseline).map(([k, v]) => (
//                       <li key={k} className="flex justify-between">
//                         <span className="text-gray-700">{k}:</span>
//                         <span className="font-medium">{fmt(v)}</span>
//                       </li>
//                     ))}
//                   </ul>
//                 </div>

//                 <div className="bg-white border rounded p-4 shadow-sm">
//                   <h5 className="font-semibold mb-2">After Mitigation</h5>
//                   <ul className="text-sm space-y-1">
//                     {Object.entries(overall).length === 0 && <li className="text-gray-500">No mitigation metrics</li>}
//                     {Object.entries(overall).map(([k, v]) => (
//                       <li key={k} className="flex justify-between">
//                         <span className="text-gray-700">{k}:</span>
//                         <span className="font-medium">
//                           {fmt(v)}
//                           {improvement(overallBaseline[k], v)}
//                         </span>
//                       </li>
//                     ))}
//                   </ul>
//                 </div>
//               </div>

//               <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
//                 <div className="bg-white border rounded p-4 shadow-sm">
//                   <h5 className="font-semibold mb-2">Performance (Before)</h5>
//                   <ul className="text-sm space-y-1">
//                     {Object.entries(perfBaseline).length === 0 && <li className="text-gray-500">No performance metrics</li>}
//                     {Object.entries(perfBaseline).map(([k, v]) => (
//                       <li key={k} className={`flex justify-between ${(perfPreference === "precision" && k === "Precision") || (perfPreference === "recall" && k === "Recall") ? "font-semibold text-green-700" : ""}`}>
//                         <span className="text-gray-700">{k}:</span>
//                         <span className="font-medium">{fmt(v)}</span>
//                       </li>
//                     ))}
//                   </ul>
//                 </div>

//                 <div className="bg-white border rounded p-4 shadow-sm">
//                   <h5 className="font-semibold mb-2">Performance (After)</h5>
//                   <ul className="text-sm space-y-1">
//                     {Object.entries(perfAfter).length === 0 && <li className="text-gray-500">No performance metrics</li>}
//                     {Object.entries(perfAfter).map(([k, v]) => (
//                       <li key={k} className={`flex justify-between ${(perfPreference === "precision" && k === "Precision") || (perfPreference === "recall" && k === "Recall") ? "font-semibold text-green-700" : ""}`}>
//                         <span className="text-gray-700">{k}:</span>
//                         <span className="font-medium">{fmt(v)}</span>
//                       </li>
//                     ))}
//                   </ul>
//                 </div>
//               </div>

//               {(Object.keys(overallBaselineTest).length > 0 || Object.keys(overallTest).length > 0) && (
//                 <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
//                   <div className="bg-white border rounded p-4 shadow-sm">
//                     <h5 className="font-semibold mb-2">Holdout Fairness (Before)</h5>
//                     <ul className="text-sm space-y-1">
//                       {Object.entries(overallBaselineTest).length === 0 && <li className="text-gray-500">No holdout metrics</li>}
//                       {Object.entries(overallBaselineTest).map(([k, v]) => (
//                         <li key={k} className="flex justify-between">
//                           <span className="text-gray-700">{k}:</span>
//                           <span className="font-medium">{fmt(v)}</span>
//                         </li>
//                       ))}
//                     </ul>
//                   </div>

//                   <div className="bg-white border rounded p-4 shadow-sm">
//                     <h5 className="font-semibold mb-2">Holdout Fairness (After)</h5>
//                     <ul className="text-sm space-y-1">
//                       {Object.entries(overallTest).length === 0 && <li className="text-gray-500">No holdout metrics</li>}
//                       {Object.entries(overallTest).map(([k, v]) => (
//                         <li key={k} className="flex justify-between">
//                           <span className="text-gray-700">{k}:</span>
//                           <span className="font-medium">{fmt(v)}</span>
//                         </li>
//                       ))}
//                     </ul>
//                   </div>
//                 </div>
//               )}

//               {(Object.keys(perfBaselineTest).length > 0 || Object.keys(perfAfterTest).length > 0) && (
//                 <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
//                   <div className="bg-white border rounded p-4 shadow-sm">
//                     <h5 className="font-semibold mb-2">Holdout Performance (Before)</h5>
//                     <ul className="text-sm space-y-1">
//                       {Object.entries(perfBaselineTest).length === 0 && <li className="text-gray-500">No holdout metrics</li>}
//                       {Object.entries(perfBaselineTest).map(([k, v]) => (
//                         <li key={k} className="flex justify-between">
//                           <span className="text-gray-700">{k}:</span>
//                           <span className="font-medium">{fmt(v)}</span>
//                         </li>
//                       ))}
//                     </ul>
//                   </div>

//                   <div className="bg-white border rounded p-4 shadow-sm">
//                     <h5 className="font-semibold mb-2">Holdout Performance (After)</h5>
//                     <ul className="text-sm space-y-1">
//                       {Object.entries(perfAfterTest).length === 0 && <li className="text-gray-500">No holdout metrics</li>}
//                       {Object.entries(perfAfterTest).map(([k, v]) => (
//                         <li key={k} className="flex justify-between">
//                           <span className="text-gray-700">{k}:</span>
//                           <span className="font-medium">{fmt(v)}</span>
//                         </li>
//                       ))}
//                     </ul>
//                   </div>
//                 </div>
//               )}

//               <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
//                 <div className="bg-white border rounded p-4 shadow-sm">
//                   <h5 className="font-semibold mb-2">Group Metrics (Before)</h5>
//                   {Object.keys(byGroupBaseline).length === 0 ? (
//                     <div className="text-gray-500 text-sm">No group metrics</div>
//                   ) : (
//                     <div className="overflow-x-auto">
//                       <table className="w-full text-sm table-auto">
//                         <thead>
//                           <tr className="text-left text-xs text-gray-600">
//                             <th className="pb-2">Group</th>
//                             <th className="pb-2">Sel.Rate</th>
//                             <th className="pb-2">FPR</th>
//                             <th className="pb-2">FNR</th>
//                           </tr>
//                         </thead>
//                         <tbody>
//                           {Object.entries(byGroupBaseline).map(([g, metrics]) => (
//                             <tr key={g} className="border-t">
//                               <td className="py-2">{g}</td>
//                               <td className="py-2">{metrics["Selection Rate"] ?? "-"}</td>
//                               <td className="py-2">{metrics["False Positive Rate"] ?? "-"}</td>
//                               <td className="py-2">{metrics["False Negative Rate"] ?? "-"}</td>
//                             </tr>
//                           ))}
//                         </tbody>
//                       </table>
//                     </div>
//                   )}
//                 </div>

//                 <div className="bg-white border rounded p-4 shadow-sm">
//                   <h5 className="font-semibold mb-2">Group Metrics (After)</h5>
//                   {Object.keys(byGroup).length === 0 ? (
//                     <div className="text-gray-500 text-sm">No group metrics</div>
//                   ) : (
//                     <div className="overflow-x-auto">
//                       <table className="w-full text-sm table-auto">
//                         <thead>
//                           <tr className="text-left text-xs text-gray-600">
//                             <th className="pb-2">Group</th>
//                             <th className="pb-2">Sel.Rate</th>
//                             <th className="pb-2">FPR</th>
//                             <th className="pb-2">FNR</th>
//                           </tr>
//                         </thead>
//                         <tbody>
//                           {Object.entries(byGroup).map(([g, metrics]) => (
//                             <tr key={g} className="border-t">
//                               <td className="py-2">{g}</td>
//                               <td className="py-2">{metrics["Selection Rate"] ?? "-"}</td>
//                               <td className="py-2">{metrics["False Positive Rate"] ?? "-"}</td>
//                               <td className="py-2">{metrics["False Negative Rate"] ?? "-"}</td>
//                             </tr>
//                           ))}
//                         </tbody>
//                       </table>
//                     </div>
//                   )}
//                 </div>
//               </div>

//               <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
//                 <div className="bg-white border rounded p-4 shadow-sm">
//                   <h5 className="font-semibold mb-2">Weights</h5>
//                   <div className="space-y-2">
//                     {(result.weights || []).map((w, i) => (
//                       <div key={i}>
//                         <div className="flex justify-between text-xs text-gray-600 mb-1">
//                           <span>Predictor {i + 1}</span>
//                           <span>{(w * 100).toFixed(1)}%</span>
//                         </div>
//                         <div className="w-full bg-gray-200 rounded-full h-2">
//                           <div className="bg-green-500 h-2 rounded-full" style={{ width: `${Math.max(0, Math.min(100, w * 100))}%` }} />
//                         </div>
//                       </div>
//                     ))}
//                   </div>
//                 </div>

//                 <div className="bg-white border rounded p-4 shadow-sm">
//                   <h5 className="font-semibold mb-2">Predictions Summary</h5>
//                   <div className="text-sm">
//                     <div className="flex justify-between">
//                       <span>Positive</span>
//                       <span className="font-medium">{posCount} ({posPct}%)</span>
//                     </div>
//                     <div className="flex justify-between">
//                       <span>Negative</span>
//                       <span className="font-medium">{negCount}</span>
//                     </div>
//                     <div className="mt-3">
//                       <div className="w-full bg-gray-200 rounded-full h-3">
//                         <div className="bg-blue-500 h-3 rounded-full" style={{ width: `${posPct}%` }} />
//                       </div>
//                     </div>
//                   </div>
//                 </div>
//               </div>
//             </>
//           )}
//         </div>
//       )}
//     </div>
//   );
// };

// export default MitigationPage;

import React, { useState, useRef, useEffect } from "react";
import { T } from "../theme";
import {
  mitigateDatasetAsync, mitigateUserModelAsync,
  getProgress, getResult,
} from "../utils/api";

/* ── shared primitives ────────────────────────────────────────────────────── */
function Card({ children, accent, style = {} }) {
  return (
    <div style={{
      background: T.surface, border: `1px solid ${T.border}`,
      borderRadius: 10, padding: "18px 20px",
      borderTop: accent ? `3px solid ${accent}` : undefined,
      ...style,
    }}>
      {children}
    </div>
  );
}

function SectionTitle({ children }) {
  return (
    <div style={{
      color: "#fff", fontSize: 13, fontWeight: 700, marginBottom: 12,
      paddingBottom: 9, borderBottom: `1px solid ${T.border}`,
    }}>
      {children}
    </div>
  );
}

function MetricCompareRow({ label, before, after }) {
  const delta = (typeof before === "number" && typeof after === "number") ? before - after : null;
  const improved = delta !== null && delta > 0;
  const worsened = delta !== null && delta < 0;
  return (
    <div style={{
      display: "flex", justifyContent: "space-between", alignItems: "center",
      padding: "8px 0", borderBottom: `1px solid ${T.border}`,
      gap: 8,
    }}>
      <span style={{ color: T.text, fontSize: 12, flex: 1 }}>{label}</span>
      <span style={{ color: T.textDim, fontFamily: "monospace", fontSize: 12, minWidth: 60, textAlign: "right" }}>
        {typeof before === "number" ? before.toFixed(4) : "—"}
      </span>
      <span style={{ color: T.textDim, fontSize: 12 }}>→</span>
      <span style={{ fontFamily: "monospace", fontSize: 12, minWidth: 60, textAlign: "right", color: T.text }}>
        {typeof after === "number" ? after.toFixed(4) : "—"}
      </span>
      {delta !== null && (
        <span style={{
          fontSize: 11, fontWeight: 700, padding: "2px 7px", borderRadius: 10,
          background: improved ? T.greenDim : worsened ? T.redDim : T.surfaceHi,
          color: improved ? T.green : worsened ? T.red : T.textDim,
          border: `1px solid ${improved ? T.green : worsened ? T.red : T.border}44`,
          minWidth: 60, textAlign: "center",
        }}>
          {improved ? "↓ " : worsened ? "↑ " : "→ "}{Math.abs(delta).toFixed(4)}
        </span>
      )}
    </div>
  );
}

function GroupTable({ title, byGroup }) {
  if (!byGroup || !Object.keys(byGroup).length) return null;
  return (
    <div style={{ background: T.surfaceHi, border: `1px solid ${T.border}`, borderRadius: 8, overflow: "hidden" }}>
      <div style={{ padding: "10px 14px", borderBottom: `1px solid ${T.border}`, color: "#fff", fontSize: 12, fontWeight: 700 }}>
        {title}
      </div>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <thead>
            <tr style={{ background: T.surface }}>
              {["Group", "Sel. Rate", "FPR", "FNR"].map(h => (
                <th key={h} style={{
                  padding: "7px 12px", textAlign: "left",
                  color: T.textDim, fontWeight: 700, fontSize: 11,
                  textTransform: "uppercase", letterSpacing: "0.04em",
                }}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.entries(byGroup).map(([g, m]) => (
              <tr key={g} style={{ borderTop: `1px solid ${T.border}` }}>
                <td style={{ padding: "7px 12px", color: T.violet, fontFamily: "monospace", fontWeight: 600 }}>{g}</td>
                <td style={{ padding: "7px 12px", color: T.text, fontFamily: "monospace" }}>{m["Selection Rate"] ?? "—"}</td>
                <td style={{ padding: "7px 12px", color: T.text, fontFamily: "monospace" }}>{m["False Positive Rate"] ?? "—"}</td>
                <td style={{ padding: "7px 12px", color: T.text, fontFamily: "monospace" }}>{m["False Negative Rate"] ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function RadioGroup({ options, value, onChange }) {
  return (
    <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
      {options.map(opt => {
        const active = value === opt.value;
        return (
          <button
            key={opt.value}
            type="button"
            onClick={() => onChange(opt.value)}
            style={{
              padding: "6px 14px", borderRadius: 6, fontSize: 12, fontWeight: 600,
              border: `1px solid ${active ? T.amber : T.border}`,
              background: active ? T.amberDim : T.surfaceHi,
              color: active ? T.amber : T.textDim,
              cursor: "pointer", fontFamily: T.font, transition: "all .15s",
            }}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}

/* ── main ─────────────────────────────────────────────────────────────────── */
const MitigationPage = ({ uploadedFile, selectedTarget, selectedSensitive, uploadedModel, isDLModel }) => {
  const [file, setFile]           = useState(uploadedFile || null);
  const [userModel, setUserModel] = useState(uploadedModel || null);
  const [target, setTarget]       = useState(selectedTarget || "");
  const [sensitive, setSensitive] = useState(selectedSensitive || "");
  const [mode, setMode]           = useState(uploadedModel ? "user_model" : "builtin");
  const [constraint, setConstraint] = useState("demographic_parity");
  const [result, setResult]       = useState(null);
  const [loading, setLoading]     = useState(false);
  const [jobPercent, setJobPercent] = useState(0);
  const [jobMessage, setJobMessage] = useState("");
  const pollRef = useRef(null);

  useEffect(() => {
    if (uploadedFile) setFile(uploadedFile);
    if (selectedTarget) setTarget(selectedTarget);
    if (selectedSensitive) setSensitive(selectedSensitive);
    if (uploadedModel) { setUserModel(uploadedModel); setMode("user_model"); }
  }, [uploadedFile, selectedTarget, selectedSensitive, uploadedModel]);

  const run = async () => {
    if (!file || !target || !sensitive) { alert("File, target and sensitive are required."); return; }
    setLoading(true); setResult(null); setJobPercent(0); setJobMessage("queued");
    try {
      const startRes = mode === "builtin"
        ? await mitigateDatasetAsync(file, target, sensitive, constraint)
        : await mitigateUserModelAsync(file, userModel, target, sensitive, constraint);

      const jobId = startRes.job_id;
      if (!jobId) { setResult(startRes); setLoading(false); return; }

      pollRef.current = setInterval(async () => {
        try {
          const p = await getProgress(jobId);
          setJobPercent(p.percent || 0);
          setJobMessage(p.message || "running");
          if (p.status === "done" || p.status === "failed") {
            clearInterval(pollRef.current);
            const final = await getResult(jobId);
            setResult(final); setLoading(false);
          }
        } catch { clearInterval(pollRef.current); setLoading(false); }
      }, 1000);
    } catch (err) {
      setResult({ error: String(err) }); setLoading(false);
    }
  };

  /* derived */
  const overallBefore = result?.metrics_baseline?.overall || {};
  const overallAfter  = result?.metrics_after_mitigation?.overall || {};
  const perfBefore    = result?.performance_baseline || {};
  const perfAfter     = result?.performance_after_mitigation || {};
  const byGroupBefore = result?.metrics_baseline?.by_group || {};
  const byGroupAfter  = result?.metrics_after_mitigation?.by_group || {};
  const suggestions   = result?.suggestions || [];
  const predictions   = result?.predictions || [];
  const posCount      = predictions.filter(p => p === 1).length;
  const posPct        = predictions.length ? ((posCount / predictions.length) * 100).toFixed(1) : "0.0";

  return (
    <div id="mitigation" style={{ maxWidth: 1100, margin: "0 auto", padding: "24px 0", fontFamily: T.font }}>

      {/* Header */}
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ color: "#fff", fontSize: 20, fontWeight: 800, margin: "0 0 4px", letterSpacing: "-0.02em" }}>
          Fairness Mitigation
        </h2>
        <div style={{ color: T.textDim, fontSize: 13 }}>
          Apply bias mitigation and compare before/after fairness metrics.
        </div>
      </div>

      {/* Config panel */}
      <Card accent={T.green} style={{ marginBottom: 16 }}>
        <SectionTitle>Configuration</SectionTitle>

        {/* Dataset summary */}
        {file && target && sensitive && (
          <div style={{
            display: "flex", gap: 20, flexWrap: "wrap", marginBottom: 16,
            padding: "10px 14px", background: T.surfaceHi,
            borderRadius: 7, border: `1px solid ${T.border}`,
          }}>
            {[
              { icon: "📄", label: "Dataset", val: file.name },
              { icon: "🎯", label: "Target", val: target },
              { icon: "👤", label: "Sensitive", val: sensitive },
              ...(mode === "user_model" && userModel ? [{ icon: "🤖", label: "Model", val: userModel.name }] : []),
            ].map(({ icon, label, val }) => (
              <div key={label} style={{ display: "flex", gap: 6, alignItems: "center" }}>
                <span style={{ fontSize: 14 }}>{icon}</span>
                <span style={{ color: T.textDim, fontSize: 12 }}>{label}:</span>
                <span style={{ color: T.text, fontSize: 12, fontWeight: 600 }}>{val}</span>
              </div>
            ))}
          </div>
        )}

        {/* Mode */}
        <div style={{ marginBottom: 14 }}>
          <div style={{ color: T.textDim, fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
            Mitigation Mode
          </div>
          <RadioGroup
            value={mode}
            onChange={setMode}
            options={[
              { value: "builtin", label: "Use baseline model from analysis" },
              { value: "user_model", label: "Use uploaded model" },
            ]}
          />
        </div>

        {/* Constraint */}
        <div style={{ marginBottom: 16 }}>
          <div style={{ color: T.textDim, fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
            Fairness Constraint
          </div>
          <RadioGroup
            value={constraint}
            onChange={setConstraint}
            options={[
              { value: "demographic_parity", label: "Demographic Parity — equal selection rates" },
              { value: "equalized_odds", label: "Equalized Odds — equal error rates" },
            ]}
          />
        </div>

        {/* DL model warning */}
        {isDLModel && (
          <div style={{
            marginBottom: 14, padding: "10px 14px", borderRadius: 7,
            background: T.amberDim, border: `1px solid ${T.amber}44`, color: T.amber, fontSize: 12,
          }}>
            ⚠ Mitigation is not available for deep-learning models (ONNX / Keras / PyTorch). You can still view fairness metrics in the report above.
          </div>
        )}

        {/* Run button */}
        <button
          onClick={run}
          disabled={loading || isDLModel}
          style={{
            padding: "10px 24px", borderRadius: 7,
            background: isDLModel ? T.surfaceHi : `linear-gradient(135deg, ${T.green}, #16a34a)`,
            border: `1px solid ${isDLModel ? T.border : "transparent"}`,
            color: isDLModel ? T.textDim : "#000",
            fontSize: 13, fontWeight: 800, cursor: isDLModel ? "not-allowed" : "pointer",
            fontFamily: T.font, opacity: loading ? 0.7 : 1, transition: "opacity .15s",
          }}
        >
          {loading ? `Running… ${jobPercent}%` : isDLModel ? "Mitigation not available" : "Run Mitigation →"}
        </button>

        {/* Progress bar */}
        {loading && (
          <div style={{ marginTop: 12 }}>
            <div style={{ color: T.textDim, fontSize: 11, marginBottom: 4 }}>{jobMessage}</div>
            <div style={{ width: "100%", height: 4, background: T.border, borderRadius: 2 }}>
              <div style={{
                width: `${jobPercent}%`, height: "100%",
                background: `linear-gradient(90deg, ${T.green}, ${T.sky})`,
                borderRadius: 2, transition: "width .4s",
              }} />
            </div>
          </div>
        )}
      </Card>

      {/* Results */}
      {result && (
        result.error ? (
          <div style={{
            background: T.redDim, border: `1px solid ${T.red}44`,
            borderRadius: 10, padding: "16px 20px", color: T.red, fontSize: 13,
          }}>
            ⚠ {result.error}
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>

            {/* Download model */}
            {result.model_download_url && (
              <div style={{
                display: "flex", alignItems: "center", gap: 12,
                padding: "12px 16px", background: T.greenDim,
                border: `1px solid ${T.green}44`, borderRadius: 8,
              }}>
                <span style={{ fontSize: 20 }}>✅</span>
                <div style={{ flex: 1 }}>
                  <div style={{ color: T.green, fontWeight: 700, fontSize: 13 }}>Mitigated model ready</div>
                  <div style={{ color: T.textDim, fontSize: 11 }}>Model ID: {result.model_id}</div>
                </div>
                <a
                  href={result.model_download_url}
                  download
                  style={{
                    padding: "7px 18px", borderRadius: 6,
                    background: T.green, color: "#000",
                    fontSize: 12, fontWeight: 800, textDecoration: "none",
                    fontFamily: T.font,
                  }}
                >
                  📥 Download
                </a>
              </div>
            )}

            {/* Suggestions */}
            {suggestions.length > 0 && (
              <Card style={{ borderLeft: `3px solid ${T.amber}`, borderRadius: "0 10px 10px 0" }}>
                <SectionTitle>⚡ Suggestions</SectionTitle>
                {suggestions.map((s, i) => (
                  <div key={i} style={{
                    display: "flex", gap: 10, padding: "7px 10px",
                    marginBottom: 6, borderRadius: 6,
                    background: T.surfaceHi, border: `1px solid ${T.border}`,
                  }}>
                    <span style={{ color: T.amber, flexShrink: 0 }}>›</span>
                    <span style={{ color: T.text, fontSize: 12, lineHeight: 1.5 }}>{s}</span>
                  </div>
                ))}
              </Card>
            )}

            {/* Before / After fairness comparison */}
            {(Object.keys(overallBefore).length > 0 || Object.keys(overallAfter).length > 0) && (
              <Card accent={T.amber}>
                <SectionTitle>Fairness Metrics — Before vs After</SectionTitle>
                <div style={{
                  display: "grid", gridTemplateColumns: "1fr auto auto auto auto",
                  gap: "0 8px", marginBottom: 6,
                  padding: "4px 0 8px",
                  borderBottom: `1px solid ${T.border}`,
                }}>
                  <span style={{ color: T.textDim, fontSize: 11, fontWeight: 700, textTransform: "uppercase" }}>Metric</span>
                  <span style={{ color: T.textDim, fontSize: 11, fontWeight: 700, minWidth: 60, textAlign: "right" }}>Before</span>
                  <span />
                  <span style={{ color: T.textDim, fontSize: 11, fontWeight: 700, minWidth: 60, textAlign: "right" }}>After</span>
                  <span style={{ color: T.textDim, fontSize: 11, fontWeight: 700, minWidth: 60, textAlign: "center" }}>Δ</span>
                </div>
                {Object.keys({ ...overallBefore, ...overallAfter }).map(k => (
                  <MetricCompareRow key={k} label={k} before={overallBefore[k]} after={overallAfter[k]} />
                ))}
              </Card>
            )}

            {/* Before / After performance comparison */}
            {(Object.keys(perfBefore).length > 0 || Object.keys(perfAfter).length > 0) && (
              <Card accent={T.sky}>
                <SectionTitle>Performance Metrics — Before vs After</SectionTitle>
                <div style={{ display: "grid", gridTemplateColumns: "1fr auto auto auto auto", gap: "0 8px", marginBottom: 6, padding: "4px 0 8px", borderBottom: `1px solid ${T.border}` }}>
                  <span style={{ color: T.textDim, fontSize: 11, fontWeight: 700, textTransform: "uppercase" }}>Metric</span>
                  <span style={{ color: T.textDim, fontSize: 11, fontWeight: 700, minWidth: 60, textAlign: "right" }}>Before</span>
                  <span />
                  <span style={{ color: T.textDim, fontSize: 11, fontWeight: 700, minWidth: 60, textAlign: "right" }}>After</span>
                  <span style={{ color: T.textDim, fontSize: 11, fontWeight: 700, minWidth: 60, textAlign: "center" }}>Δ</span>
                </div>
                {Object.keys({ ...perfBefore, ...perfAfter }).map(k => (
                  <MetricCompareRow key={k} label={k} before={perfBefore[k]} after={perfAfter[k]} />
                ))}
              </Card>
            )}

            {/* Group tables */}
            {(Object.keys(byGroupBefore).length > 0 || Object.keys(byGroupAfter).length > 0) && (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
                <GroupTable title="Group Metrics — Before" byGroup={byGroupBefore} />
                <GroupTable title="Group Metrics — After" byGroup={byGroupAfter} />
              </div>
            )}

            {/* Predictions summary + weights */}
            {(predictions.length > 0 || result.weights?.length > 0) && (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
                {predictions.length > 0 && (
                  <Card>
                    <SectionTitle>Predictions Summary</SectionTitle>
                    <div style={{ marginBottom: 10 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                        <span style={{ color: T.text, fontSize: 13 }}>Positive</span>
                        <span style={{ color: T.green, fontWeight: 700, fontSize: 13 }}>{posCount} ({posPct}%)</span>
                      </div>
                      <div style={{ width: "100%", height: 8, background: T.border, borderRadius: 4, overflow: "hidden" }}>
                        <div style={{ width: `${posPct}%`, height: "100%", background: T.green, borderRadius: 4 }} />
                      </div>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                      <span style={{ color: T.text, fontSize: 13 }}>Negative</span>
                      <span style={{ color: T.textDim, fontWeight: 700, fontSize: 13 }}>{predictions.length - posCount}</span>
                    </div>
                  </Card>
                )}

                {result.weights?.length > 0 && (
                  <Card>
                    <SectionTitle>Predictor Weights</SectionTitle>
                    {result.weights.map((w, i) => (
                      <div key={i} style={{ marginBottom: 8 }}>
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                          <span style={{ color: T.textDim, fontSize: 12 }}>Predictor {i + 1}</span>
                          <span style={{ color: T.amber, fontSize: 12, fontWeight: 700 }}>{(w * 100).toFixed(1)}%</span>
                        </div>
                        <div style={{ width: "100%", height: 5, background: T.border, borderRadius: 3, overflow: "hidden" }}>
                          <div style={{
                            width: `${Math.max(0, Math.min(100, w * 100))}%`,
                            height: "100%", background: T.amber, borderRadius: 3,
                          }} />
                        </div>
                      </div>
                    ))}
                  </Card>
                )}
              </div>
            )}
          </div>
        )
      )}
    </div>
  );
};

export default MitigationPage;
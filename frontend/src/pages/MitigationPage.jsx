import React, { useState, useRef, useEffect } from "react";
import { mitigateDataset, mitigateUserModel, mitigateDatasetAsync, mitigateUserModelAsync, getProgress, getResult } from "../utils/api";

const MitigationPage = ({ uploadedFile, selectedTarget, selectedSensitive, uploadedModel, isDLModel }) => {
  const [file, setFile] = useState(uploadedFile || null);
  const [userModel, setUserModel] = useState(uploadedModel || null);
  const [target, setTarget] = useState(selectedTarget || "");
  const [sensitive, setSensitive] = useState(selectedSensitive || "");
  const [mode, setMode] = useState(uploadedModel ? "user_model" : "builtin"); // "builtin" or "user_model"
  useEffect(() => {
    if (uploadedFile) setFile(uploadedFile);
    if (selectedTarget) setTarget(selectedTarget);
    if (selectedSensitive) setSensitive(selectedSensitive);
    if (uploadedModel) {
      setUserModel(uploadedModel);
      setMode("user_model");
    }
  }, [uploadedFile, selectedTarget, selectedSensitive, uploadedModel]);
  const [constraint, setConstraint] = useState("demographic_parity");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [jobPercent, setJobPercent] = useState(0);
  const [jobMessage, setJobMessage] = useState("");
  const pollRef = useRef(null);

  const run = async () => {
    if (!file || !target || !sensitive) { alert("Choose file, target and sensitive"); return; }
    
    setLoading(true);
    try {
      // Start async mitigation job and poll progress
      let startRes;
      if (mode === "builtin") {
        startRes = await mitigateDatasetAsync(file, target, sensitive, constraint);
      } else {
        if (!userModel) { alert("Upload your model file"); setLoading(false); return; }
        // we reuse the same async endpoint for user models for simplicity
        startRes = await mitigateUserModelAsync(file, userModel, target, sensitive, constraint);
      }

      const jobId = startRes.job_id;
      if (!jobId) {
        // fallback: maybe returned immediate result
        setResult(startRes);
        setLoading(false);
        return;
      }

      setJobPercent(0);
      setJobMessage("queued");

      // poll progress
      pollRef.current = setInterval(async () => {
        try {
          const p = await getProgress(jobId);
          setJobPercent(p.percent || 0);
          setJobMessage(p.message || "running");
          if (p.status === "done") {
            clearInterval(pollRef.current);
            const final = await getResult(jobId);
            setResult(final);
            setLoading(false);
          } else if (p.status === "failed") {
            clearInterval(pollRef.current);
            const final = await getResult(jobId);
            setResult(final);
            setLoading(false);
          }
        } catch (err) {
          console.error(err);
          clearInterval(pollRef.current);
          setLoading(false);
        }
      }, 1000);
    } catch (err) {
      console.error(err);
      setResult({ error: String(err) });
      setLoading(false);
    }
  };

  // Derived view data for nicer rendering
  const overallBaseline = result?.metrics_baseline?.overall || {};
  const overall = result?.metrics_after_mitigation?.overall || {};
  const byGroupBaseline = result?.metrics_baseline?.by_group || {};
  const byGroup = result?.metrics_after_mitigation?.by_group || {};
  const suggestions = result?.suggestions || [];
  const predictions = result?.predictions || [];
  const posCount = predictions.filter((p) => p === 1).length;
  const negCount = predictions.length - posCount;
  const posPct = predictions.length ? ((posCount / predictions.length) * 100).toFixed(1) : "0.0";
  const fmt = (v) => (typeof v === "number" ? v : v);
  const improvement = (baseline, after) => {
    if (typeof baseline === "number" && typeof after === "number") {
      const delta = baseline - after;
      const color = delta > 0 ? "text-green-600" : delta < 0 ? "text-red-600" : "text-gray-600";
      const arrow = delta > 0 ? "↓" : delta < 0 ? "↑" : "→";
      return <span className={color}> {arrow} {Math.abs(delta).toFixed(4)}</span>;
    }
    return null;
  };
  return (
    <div id="mitigation" className="max-w-4xl mx-auto mt-8 bg-white p-6 rounded-lg shadow">
      <h3 className="text-xl font-semibold mb-4">Fairness Mitigation</h3>
      
      <div className="mb-4 border-b pb-4 space-y-3">
        <div>
          <label className="block text-sm font-semibold mb-2">Mitigation Mode:</label>
          <div className="flex gap-4">
            <label className="flex items-center">
              <input type="radio" value="builtin" checked={mode === "builtin"} onChange={(e) => { setMode(e.target.value); }} className="mr-2" />
              Use baseline model from analysis
            </label>
            <label className="flex items-center">
              <input type="radio" value="user_model" checked={mode === "user_model"} onChange={(e) => { setMode(e.target.value); }} className="mr-2" />
              Use your uploaded model
            </label>
          </div>
        </div>

        <div>
          <label className="block text-sm font-semibold mb-2">Fairness Constraint:</label>
          <select value={constraint} onChange={e=>setConstraint(e.target.value)} className="border p-2 rounded">
            <option value="demographic_parity">Demographic Parity (equal selection rates)</option>
            <option value="equalized_odds">Equalized Odds (equal error rates)</option>
          </select>
        </div>
      </div>
      
      <div className="space-y-3 mb-4">
        {/* If the page was reached from Analyze, we already have file/target/sensitive; show a compact view */}
        {file && target && sensitive ? (
          <div className="text-sm text-gray-700 bg-gray-50 p-3 rounded">
            <div>📄 Data: <span className="font-medium">{file.name}</span></div>
            <div>🎯 Target: <span className="font-medium">{target}</span></div>
            <div>👤 Sensitive: <span className="font-medium">{sensitive}</span></div>
            {mode === "user_model" && userModel && (
              <div>🤖 Model: <span className="font-medium">{userModel.name}</span></div>
            )}
          </div>
        ) : (
          <>
            <div>
              <label className="block text-sm font-semibold mb-1">Data File (CSV)</label>
              <input type="file" accept=".csv" onChange={(e)=>setFile(e.target.files[0])} className="border p-2 w-full rounded" />
              {file && <p className="text-xs text-gray-600 mt-1">File: {file.name}</p>}
            </div>

            <div>
              <label className="block text-sm font-semibold mb-1">Target Column</label>
              <input placeholder="e.g., 'approved'" value={target} onChange={e=>setTarget(e.target.value)} className="border p-2 w-full rounded" />
            </div>

            <div>
              <label className="block text-sm font-semibold mb-1">Sensitive Attribute Column</label>
              <input placeholder="e.g., 'gender'" value={sensitive} onChange={e=>setSensitive(e.target.value)} className="border p-2 w-full rounded" />
            </div>
          </>
        )}
      </div>

      <button onClick={run} disabled={loading || isDLModel} className="bg-green-600 text-white px-6 py-2 rounded font-semibold hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed">
        {isDLModel ? "Mitigation not supported for deep-learning models" : loading ? "Running mitigation..." : "Run Mitigation"}
      </button>

      {isDLModel && (
        <div className="mt-3 p-3 bg-yellow-100 border border-yellow-400 text-yellow-800 rounded">
          <p className="text-sm">
            <strong>Note:</strong> Mitigation is not currently supported for deep-learning models (ONNX, Keras, PyTorch). 
            You can view the fairness analysis metrics above, but automated mitigation is available for scikit-learn and similar ML models only.
          </p>
        </div>
      )}

      {loading && (
        <div className="mt-4">
          <div className="text-sm text-gray-600 mb-2">{jobMessage} — {jobPercent}%</div>
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div className="bg-blue-600 h-3 rounded-full" style={{ width: `${jobPercent}%` }} />
          </div>
        </div>
      )}

      {result && (
        <div className="mt-6">
          <h4 className="font-semibold mb-3">Mitigation Result</h4>
          {result.error ? (
            <div className="text-red-600">{result.error}</div>
          ) : (
            <>
              {/* Suggestions Section */}
              {suggestions && suggestions.length > 0 && (
                <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 rounded-lg mb-4 shadow-sm">
                  <h5 className="text-lg font-semibold mb-2 text-yellow-800">Suggested Improvements for Dataset</h5>
                  <ul className="list-disc pl-6 text-yellow-900 space-y-1 text-sm">
                    {suggestions.map((s, i) => (
                      <li key={i}>{s}</li>
                    ))}
                  </ul>
                </div>
              )}

              {result.model_download_url && (
                <div className="mb-4">
                  <a
                    href={result.model_download_url}
                    download
                    className="bg-blue-600 text-white px-4 py-2 rounded inline-block hover:bg-blue-700"
                  >
                    📥 Download Mitigated Model
                  </a>
                  <p className="text-xs text-gray-600 mt-1">
                    Model ID: {result.model_id}
                  </p>
                </div>
              )}

              {/* Strategy / estimate */}
              {(result.strategy || result.time_estimate_seconds) && (
                <div className="mb-4 text-sm text-gray-700">
                  {result.strategy && <div>Strategy used: <span className="font-medium">{result.strategy}</span></div>}
                  {result.time_estimate_seconds && <div>Estimated time: <span className="font-medium">{result.time_estimate_seconds}s</span></div>}
                </div>
              )}

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white border rounded p-4 shadow-sm">
                  <h5 className="font-semibold mb-2">Before Mitigation</h5>
                  <ul className="text-sm space-y-1">
                    {Object.entries(overallBaseline).length === 0 && <li className="text-gray-500">No baseline metrics</li>}
                    {Object.entries(overallBaseline).map(([k, v]) => (
                      <li key={k} className="flex justify-between">
                        <span className="text-gray-700">{k}:</span>
                        <span className="font-medium">{fmt(v)}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="bg-white border rounded p-4 shadow-sm">
                  <h5 className="font-semibold mb-2">After Mitigation</h5>
                  <ul className="text-sm space-y-1">
                    {Object.entries(overall).length === 0 && <li className="text-gray-500">No mitigation metrics</li>}
                    {Object.entries(overall).map(([k, v]) => (
                      <li key={k} className="flex justify-between">
                        <span className="text-gray-700">{k}:</span>
                        <span className="font-medium">
                          {fmt(v)}
                          {improvement(overallBaseline[k], v)}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                <div className="bg-white border rounded p-4 shadow-sm">
                  <h5 className="font-semibold mb-2">Group Metrics (Before)</h5>
                  {Object.keys(byGroupBaseline).length === 0 ? (
                    <div className="text-gray-500 text-sm">No group metrics</div>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm table-auto">
                        <thead>
                          <tr className="text-left text-xs text-gray-600">
                            <th className="pb-2">Group</th>
                            <th className="pb-2">Sel.Rate</th>
                            <th className="pb-2">FPR</th>
                            <th className="pb-2">FNR</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(byGroupBaseline).map(([g, metrics]) => (
                            <tr key={g} className="border-t">
                              <td className="py-2">{g}</td>
                              <td className="py-2">{metrics["Selection Rate"] ?? "-"}</td>
                              <td className="py-2">{metrics["False Positive Rate"] ?? "-"}</td>
                              <td className="py-2">{metrics["False Negative Rate"] ?? "-"}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>

                <div className="bg-white border rounded p-4 shadow-sm">
                  <h5 className="font-semibold mb-2">Group Metrics (After)</h5>
                  {Object.keys(byGroup).length === 0 ? (
                    <div className="text-gray-500 text-sm">No group metrics</div>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm table-auto">
                        <thead>
                          <tr className="text-left text-xs text-gray-600">
                            <th className="pb-2">Group</th>
                            <th className="pb-2">Sel.Rate</th>
                            <th className="pb-2">FPR</th>
                            <th className="pb-2">FNR</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(byGroup).map(([g, metrics]) => (
                            <tr key={g} className="border-t">
                              <td className="py-2">{g}</td>
                              <td className="py-2">{metrics["Selection Rate"] ?? "-"}</td>
                              <td className="py-2">{metrics["False Positive Rate"] ?? "-"}</td>
                              <td className="py-2">{metrics["False Negative Rate"] ?? "-"}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                <div className="bg-white border rounded p-4 shadow-sm">
                  <h5 className="font-semibold mb-2">Weights</h5>
                  <div className="space-y-2">
                    {(result.weights || []).map((w, i) => (
                      <div key={i}>
                        <div className="flex justify-between text-xs text-gray-600 mb-1">
                          <span>Predictor {i + 1}</span>
                          <span>{(w * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div className="bg-green-500 h-2 rounded-full" style={{ width: `${Math.max(0, Math.min(100, w * 100))}%` }} />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-white border rounded p-4 shadow-sm">
                  <h5 className="font-semibold mb-2">Predictions Summary</h5>
                  <div className="text-sm">
                    <div className="flex justify-between">
                      <span>Positive</span>
                      <span className="font-medium">{posCount} ({posPct}%)</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Negative</span>
                      <span className="font-medium">{negCount}</span>
                    </div>
                    <div className="mt-3">
                      <div className="w-full bg-gray-200 rounded-full h-3">
                        <div className="bg-blue-500 h-3 rounded-full" style={{ width: `${posPct}%` }} />
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default MitigationPage;

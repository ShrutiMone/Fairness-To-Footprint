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
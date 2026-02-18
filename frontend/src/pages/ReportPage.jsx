// import React from "react";


// const ReportPage = ({ results }) => {
//   if (!results) return null;
//   if (results.error) {
//     return <div className="max-w-3xl mx-auto mt-8 p-4 bg-red-100 text-red-700 rounded">{results.error}</div>;
//   }
//   const { overall, by_group, suggestions, performance, data_quality, metrics_baseline_test, performance_baseline_test } = results;

//   return (
//     <div id="report" className="max-w-6xl mx-auto mt-8 space-y-6">
//       <div className="bg-white p-6 rounded-lg shadow">
//         <h3 className="text-xl font-semibold mb-4">Overall Fairness Metrics</h3>
//         <table className="w-full table-auto border-collapse">
//           <thead><tr className="bg-gray-100"><th className="p-2 text-left">Metric</th><th className="p-2 text-left">Value</th></tr></thead>
//           <tbody>
//             {overall && Object.entries(overall).map(([k,v]) => (
//               <tr key={k}><td className="border p-2">{k}</td><td className="border p-2">{(typeof v === "number") ? v.toFixed(4) : String(v)}</td></tr>
//             ))}
//           </tbody>
//         </table>
//       </div>

//       {performance && !performance.error && (
//         <div className="bg-white p-6 rounded-lg shadow">
//           <h3 className="text-xl font-semibold mb-4">Performance Metrics</h3>
//           <table className="w-full table-auto border-collapse">
//             <thead><tr className="bg-gray-100"><th className="p-2 text-left">Metric</th><th className="p-2 text-left">Value</th></tr></thead>
//             <tbody>
//               {Object.entries(performance).map(([k,v]) => (
//                 <tr key={k}><td className="border p-2">{k}</td><td className="border p-2">{(typeof v === "number") ? v.toFixed(4) : String(v)}</td></tr>
//               ))}
//             </tbody>
//           </table>
//         </div>
//       )}

//       {metrics_baseline_test && (
//         <div className="bg-white p-6 rounded-lg shadow">
//           <h3 className="text-xl font-semibold mb-4">Holdout Fairness Metrics</h3>
//           <table className="w-full table-auto border-collapse">
//             <thead><tr className="bg-gray-100"><th className="p-2 text-left">Metric</th><th className="p-2 text-left">Value</th></tr></thead>
//             <tbody>
//               {Object.entries(metrics_baseline_test.overall || {}).map(([k,v]) => (
//                 <tr key={k}><td className="border p-2">{k}</td><td className="border p-2">{(typeof v === "number") ? v.toFixed(4) : String(v)}</td></tr>
//               ))}
//             </tbody>
//           </table>
//         </div>
//       )}

//       {performance_baseline_test && !performance_baseline_test.error && (
//         <div className="bg-white p-6 rounded-lg shadow">
//           <h3 className="text-xl font-semibold mb-4">Holdout Performance Metrics</h3>
//           <table className="w-full table-auto border-collapse">
//             <thead><tr className="bg-gray-100"><th className="p-2 text-left">Metric</th><th className="p-2 text-left">Value</th></tr></thead>
//             <tbody>
//               {Object.entries(performance_baseline_test).map(([k,v]) => (
//                 <tr key={k}><td className="border p-2">{k}</td><td className="border p-2">{(typeof v === "number") ? v.toFixed(4) : String(v)}</td></tr>
//               ))}
//             </tbody>
//           </table>
//         </div>
//       )}

//       {data_quality && (
//         <div className="bg-white p-6 rounded-lg shadow">
//           <h3 className="text-xl font-semibold mb-4">Data Quality</h3>
//           <div className="text-sm text-gray-700 space-y-2">
//             <div>Rows: <span className="font-medium">{data_quality.num_rows}</span></div>
//             <div>Columns: <span className="font-medium">{data_quality.num_columns}</span></div>
//             <div>Duplicate rows: <span className="font-medium">{data_quality.duplicate_rows}</span></div>
//             {data_quality.missing_columns && Object.keys(data_quality.missing_columns).length > 0 && (
//               <div>
//                 Missing values:
//                 <ul className="list-disc pl-6">
//                   {Object.entries(data_quality.missing_columns).map(([k,v]) => (
//                     <li key={k}>{k}: {v}</li>
//                   ))}
//                 </ul>
//               </div>
//             )}
//             {data_quality.target_distribution && (
//               <div>
//                 Target distribution:
//                 <ul className="list-disc pl-6">
//                   {Object.entries(data_quality.target_distribution).map(([k,v]) => (
//                     <li key={k}>{k}: {v}</li>
//                   ))}
//                 </ul>
//               </div>
//             )}
//             {data_quality.sensitive_distribution && (
//               <div>
//                 Sensitive distribution:
//                 <ul className="list-disc pl-6">
//                   {Object.entries(data_quality.sensitive_distribution).map(([k,v]) => (
//                     <li key={k}>{k}: {v}</li>
//                   ))}
//                 </ul>
//               </div>
//             )}
//           </div>
//         </div>
//       )}

//       {/* Suggestions Section */}
//       {suggestions && suggestions.length > 0 && (
//         <div className="bg-yellow-50 border-l-4 border-yellow-400 p-6 rounded-lg shadow">
//           <h3 className="text-lg font-semibold mb-3 text-yellow-800">Suggested Improvements</h3>
//           <ul className="list-disc pl-6 text-yellow-900 space-y-2">
//             {suggestions.map((s, i) => (
//               <li key={i}>{s}</li>
//             ))}
//           </ul>
//         </div>
//       )}

//       <div className="grid md:grid-cols-2 gap-6">
//         {by_group && Object.entries(by_group).map(([group, metrics]) => (
//           <div key={group} className="bg-white p-4 rounded-lg shadow">
//             <h4 className="font-semibold text-lg mb-3">Group: {group}</h4>
//             <ul className="space-y-1">
//               {Object.entries(metrics).map(([name, val]) => (
//                 <li key={name} className="flex justify-between"><span>{name}</span><span className="font-mono">{(val === null || val === undefined) ? "N/A" : (typeof val === "number" ? val.toFixed(4) : String(val))}</span></li>
//               ))}
//             </ul>
//           </div>
//         ))}
//       </div>
//     </div>
//   );
// };

// export default ReportPage;

import React from "react";
import { T } from "../theme";

/* ── tiny shared primitives ───────────────────────────────────────────────── */
function Card({ children, accent, style = {} }) {
  return (
    <div style={{
      background: T.surface, border: `1px solid ${T.border}`,
      borderRadius: 10, padding: "20px 20px",
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
      color: "#fff", fontSize: 14, fontWeight: 700, marginBottom: 14,
      paddingBottom: 10, borderBottom: `1px solid ${T.border}`,
    }}>
      {children}
    </div>
  );
}

function MetricRow({ label, value }) {
  const num = typeof value === "number" ? value : parseFloat(value);
  const isGood = !isNaN(num) && Math.abs(num) < 0.1;
  const isBad  = !isNaN(num) && Math.abs(num) >= 0.2;
  const color  = isGood ? T.green : isBad ? T.red : T.amber;
  return (
    <div style={{
      display: "flex", justifyContent: "space-between", alignItems: "center",
      padding: "9px 0", borderBottom: `1px solid ${T.border}`,
    }}>
      <span style={{ color: T.text, fontSize: 13 }}>{label}</span>
      <span style={{
        fontFamily: "monospace", fontSize: 14, fontWeight: 700, color,
        background: color + "18", padding: "2px 10px", borderRadius: 5,
        border: `1px solid ${color}33`,
      }}>
        {typeof value === "number" ? value.toFixed(4) : String(value)}
      </span>
    </div>
  );
}

function PerfRow({ label, value }) {
  return (
    <div style={{
      display: "flex", justifyContent: "space-between", alignItems: "center",
      padding: "9px 0", borderBottom: `1px solid ${T.border}`,
    }}>
      <span style={{ color: T.text, fontSize: 13 }}>{label}</span>
      <span style={{
        fontFamily: "monospace", fontSize: 14, fontWeight: 700, color: T.sky,
        background: T.skyDim, padding: "2px 10px", borderRadius: 5,
        border: `1px solid ${T.sky}33`,
      }}>
        {typeof value === "number" ? value.toFixed(4) : String(value)}
      </span>
    </div>
  );
}

function FairnessScorecard({ overall }) {
  if (!overall) return null;
  const values = Object.values(overall).filter(v => typeof v === "number");
  if (!values.length) return null;
  const avgBias = values.reduce((s, v) => s + Math.abs(v), 0) / values.length;
  const score = Math.max(0, Math.round((1 - avgBias) * 100));
  const color = score >= 80 ? T.green : score >= 60 ? T.amber : T.red;
  const label = score >= 80 ? "FAIR" : score >= 60 ? "MODERATE BIAS" : "HIGH BIAS";

  return (
    <Card accent={color} style={{ textAlign: "center", padding: "24px 20px" }}>
      <div style={{ color: T.textDim, fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
        Overall Fairness Score
      </div>
      <div style={{ fontSize: 52, fontWeight: 900, color, lineHeight: 1, letterSpacing: "-0.04em" }}>
        {score}
      </div>
      <div style={{ fontSize: 10, color: T.textDim, marginTop: 2 }}>/ 100</div>
      <div style={{
        display: "inline-block", marginTop: 10,
        fontSize: 11, fontWeight: 700, padding: "3px 12px", borderRadius: 20,
        background: color + "22", color, border: `1px solid ${color}44`,
        letterSpacing: "0.06em",
      }}>
        {label}
      </div>
    </Card>
  );
}

function GroupCard({ group, metrics }) {
  return (
    <div style={{
      background: T.surfaceHi, border: `1px solid ${T.border}`,
      borderRadius: 8, padding: "14px 16px",
    }}>
      <div style={{
        color: T.violet, fontSize: 12, fontWeight: 700,
        fontFamily: "monospace", marginBottom: 10,
        borderBottom: `1px solid ${T.border}`, paddingBottom: 8,
      }}>
        Group: {group}
      </div>
      {Object.entries(metrics).map(([name, val]) => (
        <div key={name} style={{
          display: "flex", justifyContent: "space-between",
          padding: "5px 0", borderBottom: `1px solid ${T.border}33`,
        }}>
          <span style={{ color: T.textDim, fontSize: 12 }}>{name}</span>
          <span style={{ color: T.text, fontSize: 12, fontWeight: 600, fontFamily: "monospace" }}>
            {val === null || val === undefined ? "N/A" : typeof val === "number" ? val.toFixed(4) : String(val)}
          </span>
        </div>
      ))}
    </div>
  );
}

/* ── main ─────────────────────────────────────────────────────────────────── */
const ReportPage = ({ results }) => {
  if (!results) return null;

  if (results.error) {
    return (
      <div style={{
        maxWidth: 1100, margin: "0 auto", padding: "24px 0",
        fontFamily: T.font,
      }}>
        <div style={{
          background: T.redDim, border: `1px solid ${T.red}44`,
          borderRadius: 10, padding: "16px 20px", color: T.red,
        }}>
          ⚠ {results.error}
        </div>
      </div>
    );
  }

  const { overall, by_group, suggestions, performance, data_quality,
    metrics_baseline_test, performance_baseline_test } = results;

  const perfSource = performance || results.performance_baseline;

  return (
    <div id="report" style={{ maxWidth: 1100, margin: "0 auto", padding: "24px 0", fontFamily: T.font }}>

      {/* Report header */}
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ color: "#fff", fontSize: 20, fontWeight: 800, margin: "0 0 4px", letterSpacing: "-0.02em" }}>
          Fairness Audit Report
        </h2>
        <div style={{ color: T.textDim, fontSize: 13 }}>
          Showing bias metrics across sensitive groups.
          {results.strategy && <span> · Strategy: <code style={{ color: T.sky }}>{results.strategy}</code></span>}
        </div>
      </div>

      {/* Top row: score card + summary stats */}
      <div style={{ display: "grid", gridTemplateColumns: "220px 1fr", gap: 14, marginBottom: 14 }}>
        <FairnessScorecard overall={overall} />

        {/* Quick stat cards */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 14 }}>
          {data_quality && (
            <>
              <Card accent={T.sky}>
                <div style={{ color: T.textDim, fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em" }}>
                  Dataset Size
                </div>
                <div style={{ color: "#fff", fontSize: 28, fontWeight: 800, marginTop: 6 }}>
                  {data_quality.num_rows?.toLocaleString()}
                </div>
                <div style={{ color: T.textDim, fontSize: 11, marginTop: 2 }}>
                  rows · {data_quality.num_columns} columns
                </div>
              </Card>
              <Card accent={data_quality.duplicate_rows > 0 ? T.amber : T.green}>
                <div style={{ color: T.textDim, fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em" }}>
                  Duplicates
                </div>
                <div style={{
                  color: data_quality.duplicate_rows > 0 ? T.amber : T.green,
                  fontSize: 28, fontWeight: 800, marginTop: 6,
                }}>
                  {data_quality.duplicate_rows}
                </div>
                <div style={{ color: T.textDim, fontSize: 11, marginTop: 2 }}>
                  {Object.keys(data_quality.missing_columns || {}).length} cols with missing values
                </div>
              </Card>
            </>
          )}
        </div>
      </div>

      {/* Main metrics grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 14 }}>

        {/* Fairness metrics */}
        {overall && (
          <Card accent={T.amber}>
            <SectionTitle>Fairness Metrics</SectionTitle>
            {Object.entries(overall).map(([k, v]) => (
              <MetricRow key={k} label={k} value={v} />
            ))}
          </Card>
        )}

        {/* Performance metrics */}
        {perfSource && !perfSource.error && (
          <Card accent={T.sky}>
            <SectionTitle>Performance Metrics</SectionTitle>
            {Object.entries(perfSource).map(([k, v]) => (
              <PerfRow key={k} label={k} value={v} />
            ))}
          </Card>
        )}
      </div>

      {/* Holdout metrics */}
      {(metrics_baseline_test || performance_baseline_test) && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 14 }}>
          {metrics_baseline_test?.overall && (
            <Card>
              <SectionTitle>Holdout Fairness Metrics</SectionTitle>
              {Object.entries(metrics_baseline_test.overall).map(([k, v]) => (
                <MetricRow key={k} label={k} value={v} />
              ))}
            </Card>
          )}
          {performance_baseline_test && !performance_baseline_test.error && (
            <Card>
              <SectionTitle>Holdout Performance Metrics</SectionTitle>
              {Object.entries(performance_baseline_test).map(([k, v]) => (
                <PerfRow key={k} label={k} value={v} />
              ))}
            </Card>
          )}
        </div>
      )}

      {/* Suggestions */}
      {suggestions?.length > 0 && (
        <Card style={{ marginBottom: 14, borderLeft: `3px solid ${T.amber}`, borderRadius: "0 10px 10px 0" }}>
          <SectionTitle>⚡ Suggested Improvements</SectionTitle>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {suggestions.map((s, i) => (
              <div key={i} style={{
                display: "flex", gap: 10, alignItems: "flex-start",
                padding: "8px 12px", borderRadius: 6,
                background: T.surfaceHi, border: `1px solid ${T.border}`,
              }}>
                <span style={{ color: T.amber, fontSize: 14, flexShrink: 0, marginTop: 1 }}>›</span>
                <span style={{ color: T.text, fontSize: 13, lineHeight: 1.5 }}>{s}</span>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Data quality detail */}
      {data_quality && (
        <Card style={{ marginBottom: 14 }}>
          <SectionTitle>Data Quality</SectionTitle>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
            {data_quality.target_distribution && (
              <div>
                <div style={{ color: T.textDim, fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
                  Target Distribution
                </div>
                {Object.entries(data_quality.target_distribution).map(([k, v]) => (
                  <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", borderBottom: `1px solid ${T.border}33` }}>
                    <span style={{ color: T.text, fontSize: 12 }}>{k}</span>
                    <span style={{ color: T.sky, fontSize: 12, fontWeight: 600 }}>{v.toLocaleString()}</span>
                  </div>
                ))}
              </div>
            )}
            {data_quality.sensitive_distribution && (
              <div>
                <div style={{ color: T.textDim, fontSize: 11, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>
                  Sensitive Group Distribution
                </div>
                {Object.entries(data_quality.sensitive_distribution).map(([k, v]) => (
                  <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", borderBottom: `1px solid ${T.border}33` }}>
                    <span style={{ color: T.text, fontSize: 12 }}>{k}</span>
                    <span style={{ color: T.violet, fontSize: 12, fontWeight: 600 }}>{v.toLocaleString()}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </Card>
      )}

      {/* Group metrics */}
      {by_group && Object.keys(by_group).length > 0 && (
        <Card>
          <SectionTitle>Group-wise Metrics</SectionTitle>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 12 }}>
            {Object.entries(by_group).map(([group, metrics]) => (
              <GroupCard key={group} group={group} metrics={metrics} />
            ))}
          </div>
        </Card>
      )}
    </div>
  );
};

export default ReportPage;
import React from "react";


const ReportPage = ({ results }) => {
  if (!results) return null;
  if (results.error) {
    return <div className="max-w-3xl mx-auto mt-8 p-4 bg-red-100 text-red-700 rounded">{results.error}</div>;
  }
  const { overall, by_group, suggestions, performance, data_quality, metrics_baseline_test, performance_baseline_test } = results;

  return (
    <div id="report" className="max-w-6xl mx-auto mt-8 space-y-6">
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-xl font-semibold mb-4">Overall Fairness Metrics</h3>
        <table className="w-full table-auto border-collapse">
          <thead><tr className="bg-gray-100"><th className="p-2 text-left">Metric</th><th className="p-2 text-left">Value</th></tr></thead>
          <tbody>
            {overall && Object.entries(overall).map(([k,v]) => (
              <tr key={k}><td className="border p-2">{k}</td><td className="border p-2">{(typeof v === "number") ? v.toFixed(4) : String(v)}</td></tr>
            ))}
          </tbody>
        </table>
      </div>

      {performance && !performance.error && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-xl font-semibold mb-4">Performance Metrics</h3>
          <table className="w-full table-auto border-collapse">
            <thead><tr className="bg-gray-100"><th className="p-2 text-left">Metric</th><th className="p-2 text-left">Value</th></tr></thead>
            <tbody>
              {Object.entries(performance).map(([k,v]) => (
                <tr key={k}><td className="border p-2">{k}</td><td className="border p-2">{(typeof v === "number") ? v.toFixed(4) : String(v)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {metrics_baseline_test && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-xl font-semibold mb-4">Holdout Fairness Metrics</h3>
          <table className="w-full table-auto border-collapse">
            <thead><tr className="bg-gray-100"><th className="p-2 text-left">Metric</th><th className="p-2 text-left">Value</th></tr></thead>
            <tbody>
              {Object.entries(metrics_baseline_test.overall || {}).map(([k,v]) => (
                <tr key={k}><td className="border p-2">{k}</td><td className="border p-2">{(typeof v === "number") ? v.toFixed(4) : String(v)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {performance_baseline_test && !performance_baseline_test.error && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-xl font-semibold mb-4">Holdout Performance Metrics</h3>
          <table className="w-full table-auto border-collapse">
            <thead><tr className="bg-gray-100"><th className="p-2 text-left">Metric</th><th className="p-2 text-left">Value</th></tr></thead>
            <tbody>
              {Object.entries(performance_baseline_test).map(([k,v]) => (
                <tr key={k}><td className="border p-2">{k}</td><td className="border p-2">{(typeof v === "number") ? v.toFixed(4) : String(v)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {data_quality && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-xl font-semibold mb-4">Data Quality</h3>
          <div className="text-sm text-gray-700 space-y-2">
            <div>Rows: <span className="font-medium">{data_quality.num_rows}</span></div>
            <div>Columns: <span className="font-medium">{data_quality.num_columns}</span></div>
            <div>Duplicate rows: <span className="font-medium">{data_quality.duplicate_rows}</span></div>
            {data_quality.missing_columns && Object.keys(data_quality.missing_columns).length > 0 && (
              <div>
                Missing values:
                <ul className="list-disc pl-6">
                  {Object.entries(data_quality.missing_columns).map(([k,v]) => (
                    <li key={k}>{k}: {v}</li>
                  ))}
                </ul>
              </div>
            )}
            {data_quality.target_distribution && (
              <div>
                Target distribution:
                <ul className="list-disc pl-6">
                  {Object.entries(data_quality.target_distribution).map(([k,v]) => (
                    <li key={k}>{k}: {v}</li>
                  ))}
                </ul>
              </div>
            )}
            {data_quality.sensitive_distribution && (
              <div>
                Sensitive distribution:
                <ul className="list-disc pl-6">
                  {Object.entries(data_quality.sensitive_distribution).map(([k,v]) => (
                    <li key={k}>{k}: {v}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Suggestions Section */}
      {suggestions && suggestions.length > 0 && (
        <div className="bg-yellow-50 border-l-4 border-yellow-400 p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-3 text-yellow-800">Suggested Improvements</h3>
          <ul className="list-disc pl-6 text-yellow-900 space-y-2">
            {suggestions.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="grid md:grid-cols-2 gap-6">
        {by_group && Object.entries(by_group).map(([group, metrics]) => (
          <div key={group} className="bg-white p-4 rounded-lg shadow">
            <h4 className="font-semibold text-lg mb-3">Group: {group}</h4>
            <ul className="space-y-1">
              {Object.entries(metrics).map(([name, val]) => (
                <li key={name} className="flex justify-between"><span>{name}</span><span className="font-mono">{(val === null || val === undefined) ? "N/A" : (typeof val === "number" ? val.toFixed(4) : String(val))}</span></li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ReportPage;

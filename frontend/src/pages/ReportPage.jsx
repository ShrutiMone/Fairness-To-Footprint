import React from "react";

const ReportPage = ({ results }) => {
  if (!results) return null;
  if (results.error) {
    return <div className="max-w-3xl mx-auto mt-8 p-4 bg-red-100 text-red-700 rounded">{results.error}</div>;
  }
  const { overall, by_group } = results;

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

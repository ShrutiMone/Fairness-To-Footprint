import React from "react";

const ReportPage = ({ results }) => {
  if (!results) return null;

  // Error handling from backend
  if (results.error) {
    return (
      <div className="max-w-xl mx-auto mt-10 bg-red-100 border border-red-400 text-red-700 px-6 py-4 rounded-xl shadow">
        <h2 className="text-lg font-semibold mb-2">Error</h2>
        <p>{results.error}</p>
      </div>
    );
  }

  const { overall, by_group } = results;

  return (
    <div className="max-w-5xl mx-auto mt-10 bg-white shadow-md rounded-xl p-8">
      <h2 className="text-3xl font-semibold text-center text-blue-700 mb-8">
        Fairness Analysis Report
      </h2>

      {/* ===================== OVERALL METRICS ===================== */}
      <div className="mb-10">
        <h3 className="text-xl font-semibold text-gray-700 mb-4 border-b pb-2">
          Overall Fairness Metrics
        </h3>
        <table className="table-auto w-full border-collapse border border-gray-300 text-gray-700">
          <thead className="bg-gray-100">
            <tr>
              <th className="border border-gray-300 px-4 py-2 text-left">Metric</th>
              <th className="border border-gray-300 px-4 py-2 text-left">Value</th>
            </tr>
          </thead>
          <tbody>
            {overall &&
              Object.entries(overall).map(([metric, value]) => (
                <tr key={metric}>
                  <td className="border border-gray-300 px-4 py-2 font-medium">
                    {metric}
                  </td>
                  <td className="border border-gray-300 px-4 py-2">
                    {typeof value === "number"
                      ? value.toFixed(4)
                      : value === null
                      ? "N/A"
                      : String(value)}
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>

      {/* ===================== GROUP-WISE METRICS ===================== */}
      <div>
        <h3 className="text-xl font-semibold text-gray-700 mb-4 border-b pb-2">
          Group-wise Fairness Breakdown
        </h3>
        <div className="grid md:grid-cols-2 gap-6">
          {by_group &&
            Object.entries(by_group).map(([group, metrics]) => (
              <div
                key={group}
                className="bg-blue-50 border border-blue-200 rounded-xl p-4 shadow-sm hover:shadow-md transition"
              >
                <h4 className="font-semibold text-lg text-blue-800 mb-2">
                  Group: {group}
                </h4>
                <ul className="text-gray-700 space-y-1">
                  {Object.entries(metrics).map(([name, value]) => (
                    <li key={name} className="flex justify-between">
                      <span>{name}</span>
                      <span className="font-mono">
                        {typeof value === "number"
                          ? value.toFixed(4)
                          : String(value)}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
};

export default ReportPage;

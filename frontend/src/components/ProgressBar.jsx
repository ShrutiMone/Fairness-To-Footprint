import React from "react";

const ProgressBar = ({ step=1 }) => {
  const percent = step === 1 ? 33 : step === 2 ? 66 : 100;
  const labels = ["Upload", "Processing", "Report"];
  return (
    <div className="max-w-6xl mx-auto px-6 mt-6">
      <div className="relative bg-gray-200 h-2 rounded-full overflow-hidden">
        <div style={{width: `${percent}%`}} className="absolute left-0 top-0 h-2 bg-gradient-to-r from-green-400 to-blue-500 transition-all"/>
      </div>
      <div className="flex justify-between text-xs text-gray-600 mt-2">
        {labels.map((l,i) => <div key={i} className={i+1===step ? "font-semibold text-gray-800" : ""}>{l}</div>)}
      </div>
    </div>
  );
};

export default ProgressBar;

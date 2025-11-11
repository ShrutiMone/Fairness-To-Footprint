import React from "react";

const ProgressBar = ({ step }) => {
  const stages = ["Inputs", "Processing", "Report"];
  return (
    <div className="w-full bg-gray-200 h-3 rounded-full mt-4">
      <div
        className={`h-3 rounded-full transition-all duration-500 ${
          step === 1
            ? "w-1/3 bg-blue-500"
            : step === 2
            ? "w-2/3 bg-blue-600"
            : "w-full bg-green-500"
        }`}
      />
      <div className="flex justify-between text-sm mt-2">
        {stages.map((s, i) => (
          <span key={i} className={`${step - 1 >= i ? "text-blue-700" : "text-gray-400"}`}>
            {s}
          </span>
        ))}
      </div>
    </div>
  );
};

export default ProgressBar;

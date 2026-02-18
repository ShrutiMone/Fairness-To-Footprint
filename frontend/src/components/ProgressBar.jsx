// import React from "react";

// const ProgressBar = ({ step=1 }) => {
//   const percent = step === 1 ? 33 : step === 2 ? 66 : 100;
//   const labels = ["Upload", "Processing", "Report"];
//   return (
//     <div className="max-w-6xl mx-auto px-6 mt-6">
//       <div className="relative bg-gray-200 h-2 rounded-full overflow-hidden">
//         <div style={{width: `${percent}%`}} className="absolute left-0 top-0 h-2 bg-gradient-to-r from-green-400 to-blue-500 transition-all"/>
//       </div>
//       <div className="flex justify-between text-xs text-gray-600 mt-2">
//         {labels.map((l,i) => <div key={i} className={i+1===step ? "font-semibold text-gray-800" : ""}>{l}</div>)}
//       </div>
//     </div>
//   );
// };

// export default ProgressBar;

import React from "react";
import { T } from "../theme";

const steps = [
  { label: "Upload", num: 1 },
  { label: "Processing", num: 2 },
  { label: "Report", num: 3 },
];

const ProgressBar = ({ step = 1 }) => (
  <div style={{
    background: T.surface,
    borderBottom: `1px solid ${T.border}`,
    padding: "12px 32px",
    fontFamily: T.font,
  }}>
    <div style={{ maxWidth: 1200, margin: "0 auto", display: "flex", alignItems: "center", gap: 0 }}>
      {steps.map((s, i) => {
        const done    = step > s.num;
        const active  = step === s.num;
        const color   = done ? T.green : active ? T.amber : T.textDim;
        const isLast  = i === steps.length - 1;

        return (
          <React.Fragment key={s.num}>
            {/* Step node */}
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <div style={{
                width: 24, height: 24, borderRadius: "50%",
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 11, fontWeight: 800,
                background: done ? T.green + "22" : active ? T.amber + "22" : T.surfaceHi,
                border: `1.5px solid ${color}`,
                color,
                transition: "all .3s",
              }}>
                {done ? "✓" : s.num}
              </div>
              <span style={{
                fontSize: 12, fontWeight: active ? 700 : 500,
                color: active ? "#fff" : T.textDim,
                transition: "all .3s",
              }}>
                {s.label}
              </span>
            </div>

            {/* Connector line */}
            {!isLast && (
              <div style={{
                flex: 1, height: 1.5, margin: "0 12px",
                background: done ? T.green + "55" : T.border,
                transition: "background .3s",
              }} />
            )}
          </React.Fragment>
        );
      })}
    </div>
  </div>
);

export default ProgressBar;
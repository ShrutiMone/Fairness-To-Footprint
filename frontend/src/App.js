import React, { useState } from "react";
import Navbar from "./components/Navbar";
import ProgressBar from "./components/ProgressBar";
import FileUpload from "./components/FileUpload";
import ReportPage from "./pages/ReportPage";
import MitigationPage from "./pages/MitigationPage";
import { analyzeDataset } from "./utils/api";
import { T } from "./theme";

function App(){
  const [step, setStep] = useState(1);
  const [results, setResults] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [selectedTarget, setSelectedTarget] = useState(null);
  const [selectedSensitive, setSelectedSensitive] = useState(null);
  const [uploadedModel, setUploadedModel] = useState(null);
  const [isDLModel, setIsDLModel] = useState(false);

  const handleSubmit = async (file, target, sensitive, pred_col, train_baseline = true, modelFile = null, wrapModel = false) => {
    setUploadedFile(file);
    setSelectedTarget(target);
    setSelectedSensitive(sensitive);
    setUploadedModel(modelFile);
    setStep(2);
    const res = await analyzeDataset(file, target, sensitive, pred_col, train_baseline, modelFile, wrapModel);
    setResults(res);
    setIsDLModel(res.is_dl_model || false);
    setStep(3);
  };

  return (
    <div style={{ minHeight: "100vh", background: T.bg, fontFamily: T.font }}>
      <Navbar />
      <ProgressBar step={step} />

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0 32px" }}>

        {step === 1 && <FileUpload onSubmit={handleSubmit} />}

        {step === 2 && (
          <div style={{
            display: "flex", flexDirection: "column", alignItems: "center",
            justifyContent: "center", minHeight: "50vh", gap: 16,
          }}>
            <div style={{
              width: 40, height: 40, borderRadius: "50%",
              border: `3px solid ${T.border}`,
              borderTop: `3px solid ${T.amber}`,
              animation: "spin 0.8s linear infinite",
            }} />
            <div style={{ color: T.textDim, fontSize: 14 }}>Analysing dataset…</div>
            <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
          </div>
        )}

        {step === 3 && (
          <>
            {/* Back button */}
            <div style={{ paddingTop: 20 }}>
              <button
                onClick={() => { setStep(1); setResults(null); }}
                style={{
                  padding: "6px 14px", borderRadius: 6,
                  border: `1px solid ${T.border}`, background: T.surfaceHi,
                  color: T.textDim, fontSize: 12, cursor: "pointer",
                  fontFamily: T.font, fontWeight: 600,
                }}
              >
                ← Analyse another dataset
              </button>
            </div>

            <ReportPage results={results} />

            {/* Divider */}
            <div style={{
              margin: "32px 0", height: 1,
              background: `linear-gradient(90deg, transparent, ${T.border}, transparent)`,
            }} />

            <MitigationPage
              uploadedFile={uploadedFile}
              selectedTarget={selectedTarget}
              selectedSensitive={selectedSensitive}
              uploadedModel={uploadedModel}
              isDLModel={isDLModel}
            />

            {/* Footer padding */}
            <div style={{ height: 48 }} />
          </>
        )}
      </div>
    </div>
  );
}

export default App;

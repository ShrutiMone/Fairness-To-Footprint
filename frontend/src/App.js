import React, { useState } from "react";
import Navbar from "./components/Navbar";
import ProgressBar from "./components/ProgressBar";
import FileUpload from "./components/FileUpload";
import ReportPage from "./pages/ReportPage";
import MitigationPage from "./pages/MitigationPage";
import { analyzeDataset } from "./utils/api";

function App(){
  const [step, setStep] = useState(1);
  const [results, setResults] = useState(null);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [selectedTarget, setSelectedTarget] = useState(null);
  const [selectedSensitive, setSelectedSensitive] = useState(null);
  const [uploadedModel, setUploadedModel] = useState(null);

  const handleSubmit = async (file, target, sensitive, pred_col, train_baseline=true, modelFile=null, wrapModel=false) => {
    setUploadedFile(file);
    setSelectedTarget(target);
    setSelectedSensitive(sensitive);
    setUploadedModel(modelFile); // Store the uploaded model for use in Mitigation
    setStep(2);
    const res = await analyzeDataset(file, target, sensitive, pred_col, train_baseline, modelFile, wrapModel);
    setResults(res);
    setStep(3);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      <ProgressBar step={step} />
      <div className="max-w-6xl mx-auto px-6 py-8">
        {step === 1 && <FileUpload onSubmit={handleSubmit} />}
        {step === 2 && <div className="text-center mt-8">Processing... please wait.</div>}
        {step === 3 && (
          <>
            <ReportPage results={results} />
            <div className="mt-6">
              <MitigationPage uploadedFile={uploadedFile} selectedTarget={selectedTarget} selectedSensitive={selectedSensitive} uploadedModel={uploadedModel} />
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default App;

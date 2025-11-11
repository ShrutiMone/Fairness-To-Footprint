import React, { useState } from "react";
import FileUpload from "./components/FileUpload";
import ProgressBar from "./components/ProgressBar";
import ReportPage from "./components/ReportPage";
import { analyzeDataset } from "./utils/api";

function App() {
  const [step, setStep] = useState(1);
  const [results, setResults] = useState(null);

  const handleSubmit = async (file, target, sensitive) => {
    setStep(2);
    const res = await analyzeDataset(file, target, sensitive);
    setResults(res);
    setStep(3);
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <ProgressBar step={step} />
      {step === 1 && <FileUpload onSubmit={handleSubmit} />}
      {step === 2 && <p className="text-center mt-10 text-lg text-gray-600 animate-pulse">Processing dataset...</p>}
      {step === 3 && <ReportPage results={results} />}
    </div>
  );
}

export default App;

export const analyzeDataset = async (file, target, sensitive, pred_col=null, train_baseline=true, modelFile=null, wrapModel=false) => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("target", target);
  formData.append("sensitive", sensitive);
  formData.append("train_baseline", train_baseline ? "1" : "0");
  formData.append("wrap_model", wrapModel ? "1" : "0");
  if (pred_col) formData.append("pred_col", pred_col);
  if (modelFile) formData.append("user_model", modelFile);

  const res = await fetch("http://127.0.0.1:5000/analyze", { method: "POST", body: formData });
  return res.json();
};

export const mitigateDataset = async (file, target, sensitive, constraint="demographic_parity") => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("target", target);
  formData.append("sensitive", sensitive);
  formData.append("constraint", constraint);

  const res = await fetch("http://127.0.0.1:5000/mitigate", { method: "POST", body: formData });
  return res.json();
};

export const mitigateDatasetAsync = async (file, target, sensitive, constraint="demographic_parity") => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("target", target);
  formData.append("sensitive", sensitive);
  formData.append("constraint", constraint);

  const res = await fetch("http://127.0.0.1:5000/mitigate_async", { method: "POST", body: formData });
  return res.json(); // { job_id }
};

export const getProgress = async (jobId) => {
  const res = await fetch(`http://127.0.0.1:5000/progress/${jobId}`);
  return res.json();
};

export const getResult = async (jobId) => {
  const res = await fetch(`http://127.0.0.1:5000/result/${jobId}`);
  return res.json();
};

export const mitigateUserModel = async (dataFile, modelFile, target, sensitive, constraint="demographic_parity") => {
  const formData = new FormData();
  formData.append("file", dataFile);
  formData.append("user_model", modelFile);
  formData.append("target", target);
  formData.append("sensitive", sensitive);
  formData.append("constraint", constraint);

  const res = await fetch("http://127.0.0.1:5000/mitigate_user_model", { method: "POST", body: formData });
  return res.json();
};

export const mitigateUserModelAsync = async (dataFile, modelFile, target, sensitive, constraint="demographic_parity") => {
  const formData = new FormData();
  formData.append("file", dataFile);
  formData.append("user_model", modelFile);
  formData.append("target", target);
  formData.append("sensitive", sensitive);
  formData.append("constraint", constraint);

  const res = await fetch("http://127.0.0.1:5000/mitigate_user_model_async", { method: "POST", body: formData });
  return res.json();
};

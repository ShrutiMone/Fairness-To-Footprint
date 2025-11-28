export const analyzeDataset = async (file, target, sensitive, pred_col=null) => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("target", target);
  formData.append("sensitive", sensitive);
  if (pred_col) formData.append("pred_col", pred_col);

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

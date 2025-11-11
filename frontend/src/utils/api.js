export const analyzeDataset = async (file, target, sensitive) => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("target", target);
  formData.append("sensitive", sensitive);

  const res = await fetch("http://127.0.0.1:5000/analyze", {
  method: "POST",
  body: formData,
});

  return res.json();
};

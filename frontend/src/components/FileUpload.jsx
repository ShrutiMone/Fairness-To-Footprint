import React, { useState } from "react";

const FileUpload = ({ onSubmit }) => {
  const [file, setFile] = useState(null);
  const [target, setTarget] = useState("");
  const [sensitiveAttr, setSensitiveAttr] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (file && target && sensitiveAttr) {
      onSubmit(file, target, sensitiveAttr);
    }
  };

  return (
    <form
      className="bg-white shadow-md p-6 rounded-xl max-w-md mx-auto mt-10"
      onSubmit={handleSubmit}
    >
      <h2 className="text-xl font-semibold mb-4 text-center">Upload Dataset</h2>
      <input
        type="file"
        accept=".csv"
        onChange={(e) => setFile(e.target.files[0])}
        className="border p-2 w-full rounded mb-3"
      />
      <input
        type="text"
        placeholder="Target Variable (e.g. 'income')"
        value={target}
        onChange={(e) => setTarget(e.target.value)}
        className="border p-2 w-full rounded mb-3"
      />
      <input
        type="text"
        placeholder="Sensitive Attribute (e.g. 'gender')"
        value={sensitiveAttr}
        onChange={(e) => setSensitiveAttr(e.target.value)}
        className="border p-2 w-full rounded mb-3"
      />
      <button
        type="submit"
        className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 w-full"
      >
        Analyze Fairness
      </button>
    </form>
  );
};

export default FileUpload;

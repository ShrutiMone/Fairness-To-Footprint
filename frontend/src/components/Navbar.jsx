import React from "react";

const Navbar = () => (
  <nav className="bg-gradient-to-r from-sky-600 to-indigo-600 text-white shadow">
    <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
      <div className="flex items-center space-x-3">
        <div className="bg-white rounded-full p-1 text-indigo-600 font-bold">AI</div>
        <h1 className="text-lg font-semibold">FairCheck AI</h1>
      </div>
      <div className="flex items-center space-x-6 text-sm">
        <a href="#upload" className="hover:underline">Upload</a>
        <a href="#report" className="hover:underline">Report</a>
        <a href="#mitigation" className="hover:underline">Mitigation</a>
        <a href="https://github.com" className="hover:underline">GitHub</a>
      </div>
    </div>
  </nav>
);

export default Navbar;

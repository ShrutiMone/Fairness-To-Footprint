import React from "react";
import { T } from "../theme";

const Navbar = () => (
  <nav style={{
    background: T.surface,
    borderBottom: `1px solid ${T.border}`,
    fontFamily: T.font,
  }}>
    <div style={{
      maxWidth: 1200,
      margin: "0 auto",
      padding: "0 32px",
      height: 52,
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
    }}>
      {/* Logo + Title */}
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <div style={{
          width: 28, height: 28, borderRadius: 7,
          background: `linear-gradient(135deg, ${T.amber}, #e07b00)`,
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 14, fontWeight: 900, color: "#000",
        }}>
          ⚖
        </div>
        <span style={{ color: "#fff", fontSize: 16, fontWeight: 800, letterSpacing: "-0.02em" }}>
          FairCheck <span style={{ color: T.amber }}>AI</span>
        </span>
        <span style={{
          fontSize: 10, fontWeight: 700, letterSpacing: "0.08em",
          textTransform: "uppercase", color: T.textDim,
          borderLeft: `1px solid ${T.border}`, paddingLeft: 10, marginLeft: 2,
        }}>
          Fairness Audit System
        </span>
      </div>

      {/* Nav links */}
      <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
        {[
          { label: "Upload", href: "#upload" },
          { label: "Report", href: "#report" },
          { label: "Mitigation", href: "#mitigation" },
          { label: "GitHub", href: "https://github.com" },
        ].map(({ label, href }) => (
          <a
            key={label}
            href={href}
            style={{
              color: T.textDim,
              fontSize: 13,
              fontWeight: 600,
              textDecoration: "none",
              padding: "5px 11px",
              borderRadius: 6,
              transition: "all .15s",
            }}
            onMouseEnter={e => { e.target.style.color = T.text; e.target.style.background = T.surfaceHi; }}
            onMouseLeave={e => { e.target.style.color = T.textDim; e.target.style.background = "transparent"; }}
          >
            {label}
          </a>
        ))}
      </div>
    </div>
  </nav>
);

export default Navbar;
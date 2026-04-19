/* ── Shared UI primitives ──────────────────────────────────── */

export function Button({
  children,
  onClick,
  disabled,
  loading,
  variant = "primary",
  style: extra,
}) {
  const base = {
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    padding: "0.65rem 1.5rem",
    borderRadius: "var(--radius-sm)",
    border: "none",
    cursor: disabled || loading ? "not-allowed" : "pointer",
    fontFamily: "var(--font-mono)",
    fontSize: "0.8rem",
    fontWeight: 600,
    letterSpacing: "0.05em",
    transition: "all 0.2s ease",
    opacity: disabled || loading ? 0.5 : 1,
    ...extra,
  };
  const variants = {
    primary: {
      background: "var(--accent-lime)",
      color: "#060612",
      boxShadow: "0 0 18px rgba(184,255,63,0.2)",
    },
    secondary: {
      background: "transparent",
      color: "var(--text-primary)",
      border: "1px solid var(--border-subtle)",
    },
    danger: {
      background: "rgba(255,69,69,0.15)",
      color: "var(--accent-red)",
      border: "1px solid rgba(255,69,69,0.3)",
    },
    ghost: {
      background: "transparent",
      color: "var(--accent-cyan)",
      border: "1px solid rgba(0,229,255,0.2)",
    },
  };
  return (
    <button
      style={{ ...base, ...variants[variant] }}
      onClick={onClick}
      disabled={disabled || loading}
    >
      {loading ? <Spinner size={14} /> : null}
      {children}
    </button>
  );
}

export function Spinner({ size = 18, color = "currentColor" }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      style={{ animation: "spin 0.9s linear infinite" }}
    >
      <circle
        cx="12"
        cy="12"
        r="10"
        stroke={color}
        strokeWidth="3"
        fill="none"
        strokeDasharray="47"
        strokeDashoffset="12"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function Toggle({ label, value, onChange }) {
  return (
    <label
      style={{
        display: "flex",
        alignItems: "center",
        gap: "0.6rem",
        cursor: "pointer",
        userSelect: "none",
      }}
    >
      <div
        onClick={() => onChange(!value)}
        style={{
          width: 40,
          height: 22,
          borderRadius: 11,
          background: value ? "var(--accent-lime)" : "var(--bg-hover)",
          border: `1px solid ${value ? "transparent" : "var(--border-subtle)"}`,
          position: "relative",
          transition: "background 0.2s, border 0.2s",
          cursor: "pointer",
        }}
      >
        <div
          style={{
            position: "absolute",
            top: 3,
            left: value ? 21 : 3,
            width: 14,
            height: 14,
            borderRadius: "50%",
            background: value ? "#06060d" : "var(--text-dim)",
            transition: "left 0.2s, background 0.2s",
          }}
        />
      </div>
      <span
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: "0.78rem",
          color: "var(--text-secondary)",
        }}
      >
        {label}
      </span>
    </label>
  );
}

export function Slider({
  label,
  value,
  onChange,
  min = 0.001,
  max = 0.3,
  step = 0.001,
}) {
  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginBottom: "0.4rem",
        }}
      >
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "0.73rem",
            color: "var(--text-secondary)",
          }}
        >
          {label}
        </span>
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "0.73rem",
            color: "var(--accent-lime)",
            fontWeight: 600,
          }}
        >
          ε = {value.toFixed(3)}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{
          width: "100%",
          accentColor: "var(--accent-lime)",
          cursor: "pointer",
          height: 4,
        }}
      />
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontFamily: "var(--font-mono)",
          fontSize: "0.6rem",
          color: "var(--text-dim)",
          marginTop: "0.25rem",
        }}
      >
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}

export function Select({ label, value, onChange, options }) {
  return (
    <div>
      {label && (
        <div
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: "0.7rem",
            color: "var(--text-secondary)",
            marginBottom: "0.4rem",
            letterSpacing: "0.08em",
          }}
        >
          {label}
        </div>
      )}
      <div style={{ display: "flex", gap: "0.4rem", flexWrap: "wrap" }}>
        {options.map((opt) => (
          <button
            key={opt.value}
            onClick={() => onChange(opt.value)}
            style={{
              padding: "0.4rem 0.85rem",
              borderRadius: 6,
              border: `1px solid ${value === opt.value ? "var(--accent-lime)" : "var(--border-subtle)"}`,
              background:
                value === opt.value ? "rgba(184,255,63,0.1)" : "transparent",
              color:
                value === opt.value
                  ? "var(--accent-lime)"
                  : "var(--text-secondary)",
              fontFamily: "var(--font-mono)",
              fontSize: "0.73rem",
              fontWeight: value === opt.value ? 700 : 400,
              cursor: "pointer",
              transition: "all 0.15s",
            }}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export function SectionLabel({ children }) {
  return (
    <div
      style={{
        fontFamily: "var(--font-mono)",
        fontSize: "0.65rem",
        color: "var(--text-secondary)",
        letterSpacing: "0.13em",
        textTransform: "uppercase",
        marginBottom: "0.5rem",
      }}
    >
      {children}
    </div>
  );
}

export function Card({ children, style: extra, glow }) {
  return (
    <div
      style={{
        background: "var(--bg-card)",
        border: `1px solid ${glow ? "rgba(184,255,63,0.2)" : "var(--border-subtle)"}`,
        borderRadius: "var(--radius-lg)",
        padding: "1.5rem",
        boxShadow: glow ? "0 0 24px rgba(184,255,63,0.06)" : "none",
        ...extra,
      }}
    >
      {children}
    </div>
  );
}

export function Divider() {
  return (
    <div
      style={{
        borderTop: "1px solid var(--border-subtle)",
        margin: "1.25rem 0",
      }}
    />
  );
}

export function ResultImage({ src, label, downloadName }) {
  if (!src) return null;
  const fullSrc = src.startsWith("data:")
    ? src
    : `data:image/png;base64,${src}`;

  const handleDownload = () => {
    const a = document.createElement("a");
    a.href = fullSrc;
    a.download = downloadName || "cloaked_image.png";
    a.click();
  };

  return (
    <div
      style={{
        border: "1px solid rgba(184,255,63,0.2)",
        borderRadius: "var(--radius-md)",
        overflow: "hidden",
        position: "relative",
      }}
    >
      {label && (
        <div
          style={{
            padding: "0.4rem 0.75rem",
            background: "rgba(184,255,63,0.07)",
            fontFamily: "var(--font-mono)",
            fontSize: "0.65rem",
            color: "var(--accent-lime)",
            letterSpacing: "0.1em",
            textTransform: "uppercase",
            borderBottom: "1px solid rgba(184,255,63,0.1)",
          }}
        >
          {label}
        </div>
      )}
      <img
        src={fullSrc}
        alt={label}
        style={{
          width: "100%",
          display: "block",
          maxHeight: 300,
          objectFit: "contain",
          background: "var(--bg-deep)",
        }}
      />
      <button
        onClick={handleDownload}
        style={{
          position: "absolute",
          bottom: 8,
          right: 8,
          background: "rgba(184,255,63,0.9)",
          border: "none",
          borderRadius: 6,
          color: "#060612",
          padding: "0.35rem 0.8rem",
          fontFamily: "var(--font-mono)",
          fontSize: "0.7rem",
          fontWeight: 700,
          cursor: "pointer",
          display: "flex",
          alignItems: "center",
          gap: 5,
        }}
      >
        ↓ Download
      </button>
    </div>
  );
}

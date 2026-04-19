import { useState, useCallback } from 'react'
import DropZone from './components/DropZone'
import MetricsPanel from './components/MetricsPanel'
import {
  Button, Toggle, Slider, Select, SectionLabel,
  Card, Divider, ResultImage, Spinner,
} from './components/UI'

/* ── API helpers ─────────────────────────────────────────── */

function toBase64Strip(dataUrl) {
  return dataUrl?.split(',')[1] ?? dataUrl
}

async function callApi(endpoint, body) {
  const res = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.error || `HTTP ${res.status}`)
  }
  return res.json()
}

/* ── Static options ──────────────────────────────────────── */

const ART_METHODS = [
  { value: 'mi_fgsm', label: 'MI-FGSM' },
  { value: 'fgsm',    label: 'FGSM' },
  { value: 'pgd',     label: 'PGD' },
  { value: 'cw',      label: 'C&W L2' },
]

const FACE_METHODS = [
  { value: 'mi_fgsm', label: 'MI-FGSM' },
  { value: 'fgsm',    label: 'FGSM' },
  { value: 'pgd',     label: 'PGD' },
]

const COMPARE_METHODS = ['fgsm', 'mi_fgsm', 'pgd']

/* ═══════════════════════════════════════════════════════════
   ART CLOAK TAB
═══════════════════════════════════════════════════════════ */

function ArtCloakTab() {
  const [image,       setImage]       = useState(null)
  const [method,      setMethod]      = useState('mi_fgsm')
  const [intensity,   setIntensity]   = useState(0.02)
  const [mode,        setMode]        = useState('untargeted')
  const [ensemble,    setEnsemble]    = useState(true)
  const [targetClass, setTargetClass] = useState('')
  const [loading,     setLoading]     = useState(false)
  const [result,      setResult]      = useState(null)
  const [error,       setError]       = useState(null)

  const run = async () => {
    if (!image) return
    setLoading(true); setError(null); setResult(null)
    try {
      const data = await callApi('/art-cloak', {
        image_base64:  toBase64Strip(image.dataUrl),
        method, intensity, mode, ensemble,
        target_class:  mode === 'targeted' && targetClass ? targetClass : null,
      })
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={layout.twoCol}>
      {/* ── LEFT: controls ── */}
      <div style={layout.col}>
        <SectionLabel>Input Image</SectionLabel>
        <DropZone
          label="PNG / JPG — any size"
          value={image}
          onChange={setImage}
        />

        <Divider />

        <Select
          label="Attack Method"
          value={method}
          onChange={setMethod}
          options={ART_METHODS}
        />

        <div style={{ marginTop: '1rem' }}>
          <Slider
            label="Perturbation Intensity"
            value={intensity}
            onChange={setIntensity}
            min={0.001} max={0.1} step={0.001}
          />
        </div>

        <div style={{ marginTop: '1rem' }}>
          <SectionLabel>Attack Mode</SectionLabel>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            {['untargeted', 'targeted'].map(m => (
              <button
                key={m}
                onClick={() => setMode(m)}
                style={{
                  padding: '0.4rem 1rem',
                  borderRadius: 6,
                  border: `1px solid ${mode === m ? 'var(--accent-violet)' : 'var(--border-subtle)'}`,
                  background: mode === m ? 'rgba(139,92,246,0.12)' : 'transparent',
                  color: mode === m ? '#c4b5fd' : 'var(--text-secondary)',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.73rem',
                  fontWeight: mode === m ? 700 : 400,
                  cursor: 'pointer',
                  transition: 'all 0.15s',
                  textTransform: 'uppercase',
                  letterSpacing: '0.08em',
                }}
              >
                {m}
              </button>
            ))}
          </div>
        </div>

        {mode === 'targeted' && (
          <div style={{ marginTop: '0.85rem' }}>
            <SectionLabel>Target Class Name</SectionLabel>
            <input
              value={targetClass}
              onChange={e => setTargetClass(e.target.value)}
              placeholder="e.g. tabby cat"
              style={styles.textInput}
            />
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.63rem', color: 'var(--text-dim)', marginTop: 4 }}>
              must match a class in imagenet_classes.txt exactly
            </div>
          </div>
        )}

        <div style={{ marginTop: '1rem' }}>
          <Toggle label="Ensemble (ResNet50 + VGG16 + DenseNet121)" value={ensemble} onChange={setEnsemble} />
        </div>

        <div style={{ marginTop: '1.25rem' }}>
          <Button
            onClick={run}
            disabled={!image}
            loading={loading}
            style={{ width: '100%', padding: '0.75rem' }}
          >
            {loading ? 'Cloaking…' : '⚡ Run Art Cloak'}
          </Button>
        </div>

        {error && <ErrorBox msg={error} />}
      </div>

      {/* ── RIGHT: results ── */}
      <div style={layout.col}>
        {loading && <LoadingCard label={method} />}

        {result && !loading && (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginBottom: '0.85rem' }}>
              {image && (
                <ResultImage src={image.dataUrl} label="Original" downloadName="original.png" />
              )}
              <ResultImage
                src={result.cloaked_image}
                label="Cloaked"
                downloadName={`cloaked_${method}.png`}
              />
            </div>
            <MetricsPanel data={result.response} type="art" />
          </>
        )}

        {!result && !loading && (
          <EmptyState label="Run the attack to see results here" />
        )}
      </div>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════
   FACE CLOAK TAB
═══════════════════════════════════════════════════════════ */

function FaceCloakTab() {
  const [image,    setImage]    = useState(null)
  const [target,   setTarget]   = useState(null)
  const [method,   setMethod]   = useState('mi_fgsm')
  const [intensity,setIntensity]= useState(0.02)
  const [targeted, setTargeted] = useState(false)
  const [loading,  setLoading]  = useState(false)
  const [result,   setResult]   = useState(null)
  const [error,    setError]    = useState(null)

  const run = async () => {
    if (!image) return
    setLoading(true); setError(null); setResult(null)
    try {
      const body = {
        image_base64: toBase64Strip(image.dataUrl),
        method, intensity, targeted,
      }
      if (targeted && target) {
        body.target_image_base64 = toBase64Strip(target.dataUrl)
      }
      const data = await callApi('/face-cloak', body)
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={layout.twoCol}>
      {/* ── LEFT: controls ── */}
      <div style={layout.col}>
        <SectionLabel>Face Image</SectionLabel>
        <DropZone
          label="Upload a photo with a visible face"
          value={image}
          onChange={setImage}
        />

        <div style={{ marginTop: '1rem' }}>
          <Toggle label="Targeted identity attack" value={targeted} onChange={setTargeted} />
        </div>

        {targeted && (
          <div style={{ marginTop: '0.85rem' }}>
            <SectionLabel>Target Identity Image</SectionLabel>
            <DropZone
              label="Face to impersonate"
              value={target}
              onChange={setTarget}
            />
          </div>
        )}

        <Divider />

        <Select
          label="Attack Method"
          value={method}
          onChange={setMethod}
          options={FACE_METHODS}
        />

        <div style={{ marginTop: '1rem' }}>
          <Slider
            label="Cloaking Intensity"
            value={intensity}
            onChange={setIntensity}
            min={0.001} max={0.1} step={0.001}
          />
        </div>

        <div style={{ marginTop: '1.25rem' }}>
          <Button
            onClick={run}
            disabled={!image}
            loading={loading}
            style={{ width: '100%', padding: '0.75rem' }}
          >
            {loading ? 'Cloaking face…' : '🎭 Run Face Cloak'}
          </Button>
        </div>

        {error && <ErrorBox msg={error} />}

        <InfoBox>
          MI-FGSM gives the best transferability across face recognition systems.
          Cosine similarity below 0.85 = identity successfully hidden.
        </InfoBox>
      </div>

      {/* ── RIGHT: results ── */}
      <div style={layout.col}>
        {loading && <LoadingCard label={method} />}

        {result && !loading && (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginBottom: '0.85rem' }}>
              {image && (
                <ResultImage src={image.dataUrl} label="Original" downloadName="original_face.png" />
              )}
              <ResultImage
                src={result.cloaked_image}
                label="Cloaked Face"
                downloadName={`cloaked_face_${method}.png`}
              />
            </div>
            <MetricsPanel data={result.response} type="face" />
          </>
        )}

        {!result && !loading && (
          <EmptyState label="Face cloaking results will appear here" />
        )}
      </div>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════
   COMPARE ATTACKS TAB
═══════════════════════════════════════════════════════════ */

function CompareTab() {
  const [image,    setImage]    = useState(null)
  const [intensity,setIntensity]= useState(0.02)
  const [mode,     setMode]     = useState('untargeted')
  const [loading,  setLoading]  = useState(false)
  const [result,   setResult]   = useState(null)
  const [error,    setError]    = useState(null)

  const run = async () => {
    if (!image) return
    setLoading(true); setError(null); setResult(null)
    try {
      const data = await callApi('/compare-attacks', {
        image_base64: toBase64Strip(image.dataUrl),
        intensity, mode,
      })
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      {/* Top controls row */}
      <Card style={{ marginBottom: '1.5rem' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '1.5rem', alignItems: 'start' }}>
          <div>
            <SectionLabel>Input Image</SectionLabel>
            <DropZone label="PNG / JPG" value={image} onChange={setImage} />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <Slider
              label="Intensity (shared across all methods)"
              value={intensity}
              onChange={setIntensity}
              min={0.001} max={0.1} step={0.001}
            />
            <div>
              <SectionLabel>Mode</SectionLabel>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                {['untargeted', 'targeted'].map(m => (
                  <button
                    key={m}
                    onClick={() => setMode(m)}
                    style={{
                      padding: '0.35rem 0.85rem',
                      borderRadius: 6,
                      border: `1px solid ${mode === m ? 'var(--accent-violet)' : 'var(--border-subtle)'}`,
                      background: mode === m ? 'rgba(139,92,246,0.12)' : 'transparent',
                      color: mode === m ? '#c4b5fd' : 'var(--text-secondary)',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '0.7rem',
                      fontWeight: mode === m ? 700 : 400,
                      cursor: 'pointer',
                      transition: 'all 0.15s',
                    }}
                  >
                    {m}
                  </button>
                ))}
              </div>
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', justifyContent: 'flex-end' }}>
            <Button onClick={run} disabled={!image} loading={loading} style={{ width: '100%', padding: '0.75rem' }}>
              {loading ? 'Running 3 attacks…' : '⚔ Compare All Methods'}
            </Button>
            <InfoBox>Runs FGSM, MI-FGSM, and PGD simultaneously with the same ε for a fair comparison.</InfoBox>
          </div>
        </div>
      </Card>

      {error && <ErrorBox msg={error} />}

      {loading && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
          {COMPARE_METHODS.map(m => <LoadingCard key={m} label={m} />)}
        </div>
      )}

      {result && !loading && (
        <>
          {/* Side-by-side images */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
            {COMPARE_METHODS.map(m => {
              const d = result.comparison?.[m]
              return (
                <div key={m}>
                  <div style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.7rem',
                    color: 'var(--accent-cyan)',
                    letterSpacing: '0.1em',
                    textTransform: 'uppercase',
                    marginBottom: '0.5rem',
                    fontWeight: 700,
                  }}>
                    {m.replace('_', '-')}
                  </div>
                  <ResultImage
                    src={d?.cloaked_image}
                    label={null}
                    downloadName={`cloaked_${m}.png`}
                  />
                </div>
              )
            })}
          </div>

          {/* Metrics comparison table */}
          <CompareTable comparison={result.comparison} />
        </>
      )}

      {!result && !loading && (
        <EmptyState label="Side-by-side comparison of FGSM, MI-FGSM, and PGD will appear here" />
      )}
    </div>
  )
}

function CompareTable({ comparison }) {
  if (!comparison) return null
  const rows = [
    { key: 'psnr_db',  label: 'PSNR (dB)',  fmt: v => v?.toFixed(2) ?? '—',  highGood: true },
    { key: 'ssim',     label: 'SSIM',        fmt: v => v?.toFixed(5) ?? '—',  highGood: true },
    { key: 'linf',     label: 'L∞',          fmt: v => v?.toFixed(5) ?? '—',  highGood: false },
    { key: 'l2',       label: 'L2',          fmt: v => v?.toFixed(4) ?? '—',  highGood: false },
  ]

  return (
    <Card>
      <div style={{ overflowX: 'auto' }}>
        <table style={{
          width: '100%',
          borderCollapse: 'collapse',
          fontFamily: 'var(--font-mono)',
          fontSize: '0.78rem',
        }}>
          <thead>
            <tr>
              <th style={th()}>Metric</th>
              {COMPARE_METHODS.map(m => (
                <th key={m} style={th()}>{m.replace('_', '-')}</th>
              ))}
              <th style={th()}>Fooled?</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(row => {
              const vals = COMPARE_METHODS.map(m =>
                comparison[m]?.response?.quality_metrics?.[row.key]
              )
              const best = row.highGood ? Math.max(...vals.filter(Boolean)) : Math.min(...vals.filter(Boolean))
              return (
                <tr key={row.key}>
                  <td style={td('var(--text-secondary)')}>{row.label}</td>
                  {vals.map((v, i) => (
                    <td key={i} style={td(v === best ? 'var(--accent-lime)' : 'var(--text-primary)')}>
                      {row.fmt(v)}
                      {v === best && <span style={{ marginLeft: 4, color: 'var(--accent-lime)', fontSize: '0.65rem' }}>★</span>}
                    </td>
                  ))}
                  <td style={td()}>—</td>
                </tr>
              )
            })}
            <tr>
              <td style={td('var(--text-secondary)')}>Fooled</td>
              {COMPARE_METHODS.map(m => {
                const fooled = comparison[m]?.response?.attack_fooled
                return (
                  <td key={m} style={td(fooled ? 'var(--accent-lime)' : 'var(--accent-red)')}>
                    {fooled === undefined ? '—' : fooled ? '✓ YES' : '✗ NO'}
                  </td>
                )
              })}
              <td style={td()}>—</td>
            </tr>
            <tr>
              <td style={td('var(--text-secondary)')}>Top-1 After</td>
              {COMPARE_METHODS.map(m => {
                const cls = comparison[m]?.response?.cloaked_top3?.[0]?.class ?? '—'
                return <td key={m} style={td('var(--accent-cyan)')}>{cls}</td>
              })}
              <td style={td()}>—</td>
            </tr>
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function th() {
  return {
    padding: '0.6rem 0.85rem',
    textAlign: 'left',
    color: 'var(--text-dim)',
    fontSize: '0.65rem',
    letterSpacing: '0.1em',
    textTransform: 'uppercase',
    borderBottom: '1px solid var(--border-subtle)',
  }
}

function td(color = 'var(--text-primary)') {
  return {
    padding: '0.55rem 0.85rem',
    borderBottom: '1px solid rgba(255,255,255,0.03)',
    color,
  }
}

/* ═══════════════════════════════════════════════════════════
   SHARED SMALL COMPONENTS
═══════════════════════════════════════════════════════════ */

function ErrorBox({ msg }) {
  return (
    <div style={{
      marginTop: '1rem',
      padding: '0.75rem 1rem',
      borderRadius: 8,
      background: 'rgba(255,69,69,0.1)',
      border: '1px solid rgba(255,69,69,0.3)',
      fontFamily: 'var(--font-mono)',
      fontSize: '0.75rem',
      color: 'var(--accent-red)',
      lineHeight: 1.6,
    }}>
      <strong>Error:</strong> {msg}
    </div>
  )
}

function InfoBox({ children }) {
  return (
    <div style={{
      padding: '0.6rem 0.85rem',
      borderRadius: 8,
      background: 'rgba(0,229,255,0.05)',
      border: '1px solid rgba(0,229,255,0.12)',
      fontFamily: 'var(--font-mono)',
      fontSize: '0.68rem',
      color: 'var(--text-secondary)',
      lineHeight: 1.65,
    }}>
      {children}
    </div>
  )
}

function EmptyState({ label }) {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      minHeight: 260,
      gap: '0.75rem',
      border: '1px dashed var(--border-subtle)',
      borderRadius: 'var(--radius-lg)',
    }}>
      <div style={{ fontSize: '2.5rem', opacity: 0.3 }}>◈</div>
      <p style={{
        fontFamily: 'var(--font-mono)',
        fontSize: '0.73rem',
        color: 'var(--text-dim)',
        textAlign: 'center',
        maxWidth: 220,
        lineHeight: 1.7,
      }}>
        {label}
      </p>
    </div>
  )
}

function LoadingCard({ label }) {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      minHeight: 220,
      gap: '1rem',
      border: '1px solid var(--border-subtle)',
      borderRadius: 'var(--radius-lg)',
      background: 'var(--bg-card)',
      animation: 'pulse-glow 2s ease infinite',
    }}>
      <Spinner size={28} color="var(--accent-lime)" />
      <div style={{
        fontFamily: 'var(--font-mono)',
        fontSize: '0.72rem',
        color: 'var(--accent-lime)',
        letterSpacing: '0.12em',
        textTransform: 'uppercase',
      }}>
        running {label}
        <span style={{ animation: 'blink 1s step-end infinite' }}>_</span>
      </div>
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════
   LAYOUT CONSTANTS
═══════════════════════════════════════════════════════════ */

const layout = {
  twoCol: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1.4fr)',
    gap: '2rem',
    alignItems: 'start',
  },
  col: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.75rem',
  },
}

const styles = {
  textInput: {
    width: '100%',
    background: 'var(--bg-deep)',
    border: '1px solid var(--border-subtle)',
    borderRadius: 6,
    padding: '0.5rem 0.75rem',
    color: 'var(--text-primary)',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.78rem',
    outline: 'none',
  },
}

/* ═══════════════════════════════════════════════════════════
   APP ROOT
═══════════════════════════════════════════════════════════ */

const TABS = [
  { id: 'art',     label: '🎨 Art Cloak',       component: ArtCloakTab },
  { id: 'face',    label: '🎭 Face Cloak',       component: FaceCloakTab },
  { id: 'compare', label: '⚔  Compare Attacks',  component: CompareTab },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('art')
  const ActiveComponent = TABS.find(t => t.id === activeTab).component

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg-void)' }}>
      {/* ── Scanline overlay ── */}
      <div style={{
        position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 0,
        background: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.03) 2px, rgba(0,0,0,0.03) 4px)',
      }} />

      {/* ── Header ── */}
      <header style={{
        position: 'sticky', top: 0, zIndex: 50,
        background: 'rgba(6,6,13,0.85)',
        backdropFilter: 'blur(20px)',
        borderBottom: '1px solid var(--border-subtle)',
      }}>
        <div style={{
          maxWidth: 1200, margin: '0 auto',
          padding: '0 2rem',
          display: 'flex',
          alignItems: 'center',
          gap: '2rem',
          height: 62,
        }}>
          {/* Logo */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexShrink: 0 }}>
            <div style={{
              width: 32, height: 32,
              background: 'linear-gradient(135deg, var(--accent-lime), var(--accent-cyan))',
              borderRadius: 8,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '1.1rem',
              boxShadow: '0 0 16px rgba(184,255,63,0.35)',
            }}>
              ◈
            </div>
            <div>
              <div style={{
                fontFamily: 'var(--font-display)',
                fontWeight: 800,
                fontSize: '1.05rem',
                letterSpacing: '-0.02em',
                lineHeight: 1,
                background: 'linear-gradient(90deg, var(--accent-lime), var(--accent-cyan))',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}>
                MIRAGE-AI
              </div>
              <div style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.58rem',
                color: 'var(--text-dim)',
                letterSpacing: '0.15em',
                textTransform: 'uppercase',
              }}>
                v2 · adversarial cloaking
              </div>
            </div>
          </div>

          {/* Nav tabs */}
          <nav style={{ display: 'flex', gap: '0.25rem' }}>
            {TABS.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{
                  padding: '0.45rem 1rem',
                  borderRadius: 7,
                  border: 'none',
                  background: activeTab === tab.id ? 'rgba(184,255,63,0.1)' : 'transparent',
                  color: activeTab === tab.id ? 'var(--accent-lime)' : 'var(--text-secondary)',
                  fontFamily: 'var(--font-display)',
                  fontSize: '0.83rem',
                  fontWeight: activeTab === tab.id ? 700 : 400,
                  cursor: 'pointer',
                  transition: 'all 0.15s',
                  letterSpacing: '-0.01em',
                  outline: `${activeTab === tab.id ? '1px' : '0px'} solid rgba(184,255,63,0.2)`,
                }}
              >
                {tab.label}
              </button>
            ))}
          </nav>

          {/* Status badge */}
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{
              width: 6, height: 6, borderRadius: '50%',
              background: 'var(--accent-lime)',
              boxShadow: '0 0 8px var(--accent-lime)',
              animation: 'pulse-glow 2s ease infinite',
            }} />
            <span style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '0.65rem',
              color: 'var(--text-secondary)',
              letterSpacing: '0.08em',
            }}>
              API :8080
            </span>
          </div>
        </div>
      </header>

      {/* ── Hero strip ── */}
      <div style={{
        borderBottom: '1px solid var(--border-subtle)',
        background: 'linear-gradient(180deg, rgba(184,255,63,0.03) 0%, transparent 100%)',
        padding: '1.5rem 2rem',
      }}>
        <div style={{ maxWidth: 1200, margin: '0 auto' }}>
          <HeroMeta activeTab={activeTab} />
        </div>
      </div>

      {/* ── Main content ── */}
      <main style={{ maxWidth: 1200, margin: '0 auto', padding: '2rem', position: 'relative', zIndex: 1 }}>
        <div className="fade-up">
          <ActiveComponent />
        </div>
      </main>

      {/* ── Footer ── */}
      <footer style={{
        borderTop: '1px solid var(--border-subtle)',
        padding: '1.25rem 2rem',
        maxWidth: 1200,
        margin: '2rem auto 0',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '0.5rem',
      }}>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.63rem', color: 'var(--text-dim)' }}>
          Mirage-AI v2 · MI-FGSM + Ensemble · ResNet50 / VGG16 / DenseNet121 · FaceNet VGGFace2
        </span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.63rem', color: 'var(--text-dim)' }}>
          C&W L2 · PGD · PSNR · SSIM
        </span>
      </footer>
    </div>
  )
}

/* ── Hero metadata per tab ── */
function HeroMeta({ activeTab }) {
  const meta = {
    art: {
      title: 'Art Cloak',
      desc:  'Add imperceptible adversarial perturbations to fool image classifiers.',
      pills: ['FGSM', 'MI-FGSM', 'PGD', 'C&W L2', 'Ensemble', 'Targeted / Untargeted'],
    },
    face: {
      title: 'Face Cloak',
      desc:  'Perturb face images in embedding space to defeat facial recognition.',
      pills: ['FaceNet VGGFace2', 'MI-FGSM', 'FGSM', 'PGD', 'Cosine Similarity', 'Identity Transfer'],
    },
    compare: {
      title: 'Compare Attacks',
      desc:  'Run FGSM, MI-FGSM, and PGD with the same ε and compare quality metrics side-by-side.',
      pills: ['PSNR', 'SSIM', 'L∞', 'L2', 'Fooling Rate', 'Same ε'],
    },
  }
  const m = meta[activeTab]
  return (
    <div>
      <div style={{
        display: 'flex', alignItems: 'baseline', gap: '0.75rem', marginBottom: '0.4rem', flexWrap: 'wrap',
      }}>
        <h1 style={{
          fontFamily: 'var(--font-display)',
          fontWeight: 800,
          fontSize: '1.3rem',
          letterSpacing: '-0.03em',
          color: 'var(--text-primary)',
        }}>
          {m.title}
        </h1>
        <p style={{
          fontFamily: 'var(--font-mono)',
          fontSize: '0.75rem',
          color: 'var(--text-secondary)',
        }}>
          {m.desc}
        </p>
      </div>
      <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
        {m.pills.map(p => (
          <span key={p} style={{
            padding: '0.2rem 0.6rem',
            borderRadius: 4,
            background: 'var(--bg-surface)',
            border: '1px solid var(--border-subtle)',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.62rem',
            color: 'var(--text-secondary)',
            letterSpacing: '0.05em',
          }}>
            {p}
          </span>
        ))}
      </div>
    </div>
  )
}

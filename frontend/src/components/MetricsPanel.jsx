const s = {
  panel: {
    background: 'var(--bg-deep)',
    border: '1px solid var(--border-subtle)',
    borderRadius: 'var(--radius-md)',
    padding: '1.25rem',
    fontFamily: 'var(--font-mono)',
  },
  title: {
    fontSize: '0.7rem',
    color: 'var(--text-secondary)',
    letterSpacing: '0.12em',
    textTransform: 'uppercase',
    marginBottom: '1rem',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '0.6rem',
  },
  cell: {
    background: 'var(--bg-surface)',
    borderRadius: 8,
    padding: '0.6rem 0.75rem',
  },
  cellKey: {
    fontSize: '0.65rem',
    color: 'var(--text-secondary)',
    marginBottom: '0.2rem',
  },
  cellVal: (good) => ({
    fontSize: '0.95rem',
    fontWeight: 600,
    color: good === true
      ? 'var(--accent-lime)'
      : good === false
      ? 'var(--accent-red)'
      : 'var(--accent-cyan)',
  }),
  badge: (success) => ({
    display: 'inline-flex',
    alignItems: 'center',
    gap: 6,
    padding: '0.35rem 0.85rem',
    borderRadius: 99,
    fontSize: '0.75rem',
    fontWeight: 700,
    background: success ? 'rgba(184,255,63,0.1)' : 'rgba(255,69,69,0.1)',
    border: `1px solid ${success ? 'rgba(184,255,63,0.3)' : 'rgba(255,69,69,0.3)'}`,
    color: success ? 'var(--accent-lime)' : 'var(--accent-red)',
    marginBottom: '1rem',
  }),
  top3: {
    marginTop: '1rem',
  },
  top3Title: {
    fontSize: '0.65rem',
    color: 'var(--text-secondary)',
    letterSpacing: '0.1em',
    textTransform: 'uppercase',
    marginBottom: '0.5rem',
  },
  pred: (i) => ({
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '0.35rem 0.6rem',
    marginBottom: '0.3rem',
    borderRadius: 6,
    background: i === 0 ? 'rgba(0,229,255,0.07)' : 'transparent',
    border: i === 0 ? '1px solid rgba(0,229,255,0.15)' : '1px solid transparent',
  }),
  predClass: (i) => ({
    fontSize: '0.75rem',
    color: i === 0 ? 'var(--accent-cyan)' : 'var(--text-secondary)',
    fontWeight: i === 0 ? 600 : 400,
  }),
  predProb: (i) => ({
    fontSize: '0.73rem',
    color: i === 0 ? 'var(--accent-cyan)' : 'var(--text-dim)',
  }),
  bar: (prob) => ({
    height: 3,
    background: `linear-gradient(90deg, var(--accent-cyan) ${prob * 100}%, var(--bg-hover) ${prob * 100}%)`,
    borderRadius: 2,
    marginTop: 4,
  }),
  divider: {
    borderTop: '1px solid var(--border-subtle)',
    margin: '0.85rem 0',
  }
}

export function MetricCell({ label, value, good }) {
  return (
    <div style={s.cell}>
      <div style={s.cellKey}>{label}</div>
      <div style={s.cellVal(good)}>{value}</div>
    </div>
  )
}

export function Top3({ label, preds }) {
  if (!preds?.length) return null
  return (
    <div style={s.top3}>
      <div style={s.top3Title}>{label}</div>
      {preds.map((p, i) => (
        <div key={i} style={s.pred(i)}>
          <div>
            <div style={s.predClass(i)}>{p.class}</div>
            <div style={s.bar(p.prob)} />
          </div>
          <div style={s.predProb(i)}>{(p.prob * 100).toFixed(1)}%</div>
        </div>
      ))}
    </div>
  )
}

export default function MetricsPanel({ data, type = 'art' }) {
  if (!data) return null
  const q = data.quality_metrics || {}

  if (type === 'art') {
    const fooled = data.attack_fooled
    return (
      <div style={s.panel}>
        <div style={s.title}>Attack Results</div>
        <div style={s.badge(fooled)}>
          {fooled ? '✓ FOOLED' : '✗ NOT FOOLED'}
        </div>
        <div style={s.grid}>
          <MetricCell label="PSNR (dB)" value={q.psnr_db ?? '—'} good={q.psnr_db > 30} />
          <MetricCell label="SSIM" value={q.ssim ?? '—'} good={q.ssim > 0.9} />
          <MetricCell label="L∞ norm" value={q.linf?.toFixed(5) ?? '—'} />
          <MetricCell label="L2 norm" value={q.l2?.toFixed(4) ?? '—'} />
        </div>
        <div style={s.divider} />
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
          <Top3 label="Before" preds={data.original_top3} />
          <Top3 label="After" preds={data.cloaked_top3} />
        </div>
        {data.method && (
          <div style={{ marginTop: '0.75rem', fontSize: '0.67rem', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
            method: {data.method} · mode: {data.mode} · ensemble: {data.ensemble ? 'on' : 'off'}
          </div>
        )}
      </div>
    )
  }

  // Face metrics
  const success = data.attack_success
  return (
    <div style={s.panel}>
      <div style={s.title}>Face Cloaking Results</div>
      <div style={s.badge(success)}>
        {success ? '✓ IDENTITY HIDDEN' : '✗ CLOAKING WEAK'}
      </div>
      <div style={s.grid}>
        <MetricCell label="Cosine Sim After" value={data.cosine_similarity_after?.toFixed(4) ?? '—'} good={data.cosine_similarity_after < 0.85} />
        <MetricCell label="Similarity Drop" value={data.similarity_drop?.toFixed(4) ?? '—'} good={data.similarity_drop > 0.15} />
        <MetricCell label="Embed L2 Dist" value={data.embedding_l2_distance?.toFixed(4) ?? '—'} good={data.embedding_l2_distance > 0.5} />
        <MetricCell label="Cloak Score" value={data.effective_cloaking_score?.toFixed(4) ?? '—'} good={data.effective_cloaking_score > 0.5} />
      </div>
      <div style={s.divider} />
      <div style={s.grid}>
        <MetricCell label="PSNR (dB)" value={q.psnr_db ?? '—'} good={q.psnr_db > 30} />
        <MetricCell label="SSIM" value={q.ssim ?? '—'} good={q.ssim > 0.9} />
      </div>
      {data.target_similarity_after !== undefined && (
        <>
          <div style={s.divider} />
          <div style={{ fontSize: '0.68rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', lineHeight: 1.8 }}>
            <div>target sim before → {data.target_similarity_before?.toFixed(4)}</div>
            <div>target sim after  → {data.target_similarity_after?.toFixed(4)}</div>
            <div>push toward target → {data.push_toward_target?.toFixed(4)}</div>
          </div>
        </>
      )}
      <div style={{ marginTop: '0.75rem', fontSize: '0.67rem', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
        method: {data.method}
      </div>
    </div>
  )
}

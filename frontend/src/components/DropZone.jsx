import { useState, useRef, useCallback } from 'react'

const styles = {
  zone: (dragging, hasImage) => ({
    border: `2px dashed ${dragging ? 'var(--accent-lime)' : hasImage ? 'rgba(184,255,63,0.4)' : 'var(--border-subtle)'}`,
    borderRadius: 'var(--radius-lg)',
    background: dragging
      ? 'rgba(184,255,63,0.05)'
      : hasImage
      ? 'rgba(184,255,63,0.03)'
      : 'var(--bg-deep)',
    padding: hasImage ? '0' : '2.5rem 1.5rem',
    cursor: 'pointer',
    transition: 'all 0.25s ease',
    position: 'relative',
    overflow: 'hidden',
    minHeight: hasImage ? 0 : 160,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '0.75rem',
  }),
  preview: {
    width: '100%',
    maxHeight: 280,
    objectFit: 'contain',
    borderRadius: 'var(--radius-md)',
    display: 'block',
  },
  icon: (dragging) => ({
    fontSize: '2.4rem',
    filter: dragging ? 'brightness(1.5)' : 'none',
    transition: 'filter 0.2s',
  }),
  label: {
    fontFamily: 'var(--font-mono)',
    fontSize: '0.78rem',
    color: 'var(--text-secondary)',
    textAlign: 'center',
    lineHeight: 1.7,
  },
  highlight: { color: 'var(--accent-lime)', fontWeight: 600 },
  clearBtn: {
    position: 'absolute',
    top: 8, right: 8,
    background: 'rgba(255,69,69,0.15)',
    border: '1px solid rgba(255,69,69,0.3)',
    borderRadius: 6,
    color: 'var(--accent-red)',
    cursor: 'pointer',
    padding: '3px 10px',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.7rem',
    transition: 'all 0.2s',
  },
}

export default function DropZone({ label = 'Drop image here', value, onChange }) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef()

  const handleFile = useCallback((file) => {
    if (!file || !file.type.startsWith('image/')) return
    const reader = new FileReader()
    reader.onload = (e) => onChange({ dataUrl: e.target.result, file })
    reader.readAsDataURL(file)
  }, [onChange])

  const onDrop = useCallback((e) => {
    e.preventDefault(); setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }, [handleFile])

  const onDragOver = (e) => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)
  const onClick = () => inputRef.current?.click()

  const hasImage = !!value?.dataUrl

  return (
    <div
      style={styles.zone(dragging, hasImage)}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onClick={!hasImage ? onClick : undefined}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={(e) => handleFile(e.target.files[0])}
      />

      {hasImage ? (
        <>
          <img src={value.dataUrl} alt="uploaded" style={styles.preview} />
          <button
            style={styles.clearBtn}
            onClick={(e) => { e.stopPropagation(); onChange(null) }}
          >
            ✕ clear
          </button>
        </>
      ) : (
        <>
          <div style={styles.icon(dragging)}>
            {dragging ? '⬇' : '🖼'}
          </div>
          <p style={styles.label}>
            <span style={styles.highlight}>Drop image</span> or{' '}
            <span style={styles.highlight}>click to browse</span>
            <br />
            {label}
          </p>
        </>
      )}
    </div>
  )
}

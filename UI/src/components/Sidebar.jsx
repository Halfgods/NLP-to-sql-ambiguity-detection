import { useState } from 'react'

function ModelStatus({ health }) {
  const model = health?.model
  const state = model?.state || 'idle'
  const labels = {
    idle: 'Connecting to model…',
    starting: 'Starting model worker…',
    loading: 'Loading Qwen weights…',
    ready: `Qwen ready${model?.info?.device ? ` · ${model.info.device}` : ''}${model?.info?.adapter ? ' · LoRA' : ''}`,
    error: 'Model unavailable',
  }
  const detail = model?.error || model?.info?.model || ''
  return (
    <div className={`model-status ${state}`} title={detail || labels[state]}>
      <span className="model-dot" />
      <span>{labels[state] || state}</span>
      {health?.workspace?.tables != null && (
        <span className="model-tables">{health.workspace.tables} tbl</span>
      )}
    </div>
  )
}

export default function Sidebar({ datasets, schema, uploadsDir, note, uploading, health, onPick, onRemove, onReset }) {
  const [open, setOpen] = useState(true)

  return (
    <aside className={`sidebar ${open ? '' : 'collapsed'}`}>
      <button className="sidebar-toggle" onClick={() => setOpen((o) => !o)} aria-label="Toggle sidebar">
        {open ? '⟨' : '⟩'}
      </button>

      <div className="sidebar-inner">
        <div className="brand">
          <div className="brand-mark">◈</div>
          <div>
            <div className="brand-name">Clarion</div>
            <div className="brand-sub">NL → SQL, clarified first</div>
          </div>
        </div>

        <ModelStatus health={health} />

        <section className="panel">
          <h2>Datasets</h2>
          <p className="hint">
            CSV / TSV / JSON or SQLite — imported tables live in the persistent
            workspace (<code>{uploadsDir}/</code>); source files stay unchanged.
          </p>

          <div
            className={`dropzone ${uploading ? 'busy' : ''}`}
            onClick={onPick}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === 'Enter' && onPick?.()}
          >
            <div className="dz-icon">+</div>
            <div className="dz-text">{uploading ? 'Uploading…' : 'Drop a file or click to browse'}</div>
          </div>

          {note && <div className={`note ${note.ok ? 'ok' : 'err'}`}>{note.text}</div>}

          <ul className="file-list">
            {datasets.length === 0 && (
              <li className="empty">No uploads yet — the built-in <b>students</b> demo table is ready to query.</li>
            )}
            {datasets.map((d) => (
              <li key={d.stored} className="file-row">
                <div className="file-main">
                  <span className="file-name" title={d.original}>{d.original}</span>
                  <span className="file-meta">
                    {d.rows != null ? `${d.rows} rows · ` : ''}
                    {(d.size / 1024).toFixed(1)} KB
                  </span>
                </div>
                <div className="file-table">
                  {d.tables?.length > 1
                    ? `${d.tables.length} tables`
                    : <>as <code>{d.table}</code></>}
                </div>
                <button className="ghost-btn" title="Delete file" onClick={() => onRemove(d.stored)}>✕</button>
              </li>
            ))}
          </ul>

          {datasets.length > 0 && (
            <button className="ghost-btn wide" onClick={onReset}>Clear all uploads</button>
          )}
        </section>

        <section className="panel">
          <h2>Schema</h2>
          <div className="schema-list">
            {schema.map((t) => (
              <details key={`${t.source}-${t.name}`} className={`table-card src-${t.source}`}>
                <summary>
                  <span className="tbl-badge">{t.source === 'demo' ? 'demo' : 'upload'}</span>
                  <span className="tbl-name">{t.name}</span>
                  <span className="tbl-rows">{t.rows ?? '—'} rows</span>
                </summary>
                <ul>
                  {(t.columns || []).map((c) => (
                    <li key={c.name}>
                      <span className="col-name">{c.name}</span>
                      <span className={`col-type t-${c.type}`}>{c.type}</span>
                    </li>
                  ))}
                  {(t.columns || []).length === 0 && <li className="empty">no columns</li>}
                </ul>
              </details>
            ))}
          </div>
        </section>

        <footer className="sidebar-foot">
          Qwen · clarify → generate → validate → execute
        </footer>
      </div>
    </aside>
  )
}

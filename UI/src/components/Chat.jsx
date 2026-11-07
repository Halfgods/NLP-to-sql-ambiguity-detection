import { forwardRef, useState } from 'react'

/* tiny markdown-ish renderer: **bold**, `code`, paragraphs */
function RichText({ text }) {
  const blocks = String(text).split('\n').filter((l) => l.length)
  return (
    <div className="rich">
      {blocks.map((line, i) => (
        <p key={i}>
          {line.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((part, j) => {
            if (part.startsWith('**')) return <strong key={j}>{part.slice(2, -2)}</strong>
            if (part.startsWith('`')) return <code key={j}>{part.slice(1, -1)}</code>
            return <span key={j}>{part}</span>
          })}
        </p>
      ))}
    </div>
  )
}

export function SqlCard({ sql, summary }) {
  const [copied, setCopied] = useState(false)
  return (
    <div className="sql-card">
      <div className="sql-head">
        <span className="sql-label">SQL</span>
        {summary && <span className="sql-summary">{summary}</span>}
        <button
          className="ghost-btn"
          onClick={() => {
            navigator.clipboard?.writeText(sql)
            setCopied(true)
            setTimeout(() => setCopied(false), 1400)
          }}
        >
          {copied ? 'copied ✓' : 'copy'}
        </button>
      </div>
      <pre><code>{sql}</code></pre>
    </div>
  )
}

export function ToolTrace({ tools }) {
  if (!tools?.length) return null
  const schemaChecks = tools.filter((call) =>
    String(call.arguments || '').startsWith('describe_table('))
  return (
    <div className="tool-trace">
      {schemaChecks.length > 0 && (
        <div className="schema-inspected">
          <span className="tool-trace-label">Schema checked</span>
          <span>{schemaChecks.length} table{schemaChecks.length === 1 ? '' : 's'} supplied to the model</span>
        </div>
      )}
      <details>
        <summary>
          <span className="tool-trace-label">Pipeline steps</span>
          <span className="tool-trace-count">{tools.length}</span>
        </summary>
        <ol>
          {tools.map((call, i) => (
            <li key={i} className={call.ok === false ? 'tool-error' : ''}>
              <code>{call.arguments}</code>
              <pre>{call.observation}</pre>
            </li>
          ))}
        </ol>
      </details>
    </div>
  )
}

export function ResultsTable({ results }) {
  if (!results) return null
  const { columns = [], rows = [] } = results
  if (!rows.length) {
    return (
      <div className="results">
        <div className="results-head">Result · 0 rows</div>
        <p className="empty-result">0 rows returned.</p>
      </div>
    )
  }
  return (
    <div className="results">
      <div className="results-head">
        Result · {rows.length} row{rows.length === 1 ? '' : 's'}
        <span className="cols-count">{columns.length} col{columns.length === 1 ? '' : 's'}</span>
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>{columns.map((c) => <th key={c}>{c}</th>)}</tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                {columns.map((c) => {
                  const v = r[c]
                  const num = typeof v === 'number'
                  return (
                    <td key={c} className={num ? 'num' : ''}>
                      {v === null || v === undefined ? <span className="null">NULL</span> : String(v)}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export function ClarifyCard({ message, onResolve }) {
  const { ambiguities = [], resolved } = message
  return (
    <div className={`clarify${resolved ? ' resolved' : ''}`}>
      {resolved ? (
        <div className="clarify-title">✓ Clarified — {resolved}</div>
      ) : (
        <div className="clarify-title">
          <span className="amb-icon">？</span>
          {ambiguities.length > 1 ? `${ambiguities.length} ambiguities detected` : 'Ambiguity detected'}
        </div>
      )}
      {ambiguities.map((amb) => (
        <div key={amb.id} className="amb-block">
          {!resolved && <div className="amb-question">{amb.question}</div>}
          <div className="chips">
            {amb.options.map((opt, i) => {
              const label = String(opt.label)
              const selected = Boolean(resolved) && label === resolved
              return (
                <button
                  key={i}
                  className={`chip${selected ? ' selected' : ''}`}
                  onClick={() => onResolve?.(amb.id, i, label)}
                  disabled={Boolean(resolved)}
                  title={opt.note || ''}
                >
                  <span className="chip-label">{label}</span>
                  {opt.note && <span className="chip-note">{opt.note}</span>}
                </button>
              )
            })}
          </div>
        </div>
      ))}
      {!resolved && !onResolve && (
        <p className="hint-sm">
          {ambiguities.some((a) => (a.options || []).length > 0)
            ? 'Pick an option above 👆'
            : 'Reply below with what you meant 👇'}
        </p>
      )}
    </div>
  )
}

const Chat = forwardRef(function Chat({ messages, busy, pendingId, onResolve, onConfirmWrite, onCancelWrite }, ref) {
  return (
    <div className="chat" ref={ref}>
      <div className="messages">
        {messages.map((m) => (
          <article key={m.id} className={`msg ${m.role}`}>
            <div className="bubble-wrap">
              <div className={`bubble ${m.role} ${m.kind}`}>
                <RichText text={m.text} />
                {m.kind === 'progress' && (
                  <div className="progress-caption">
                    <span className="dot" />
                    {m.progressLabel || 'Working…'}
                  </div>
                )}
                {m.tools?.length > 0 && <ToolTrace tools={m.tools} />}
                {m.explanation && <p className="query-explanation">{m.explanation}</p>}
                {m.sql && <SqlCard sql={m.sql} summary={m.summary} />}
                {m.kind === 'write_confirmation' && (
                  <div className="write-confirm">
                    <p>Rollback-only preview: {m.affectedRows} row{m.affectedRows === 1 ? '' : 's'} would change. No data has changed yet.</p>
                    {m.previewDiff?.columns?.length > 0 && (m.previewDiff.before.length > 0 || m.previewDiff.after.length > 0) && (
                      <div className="write-preview-diff">
                        <table>
                          <thead>
                            <tr>
                              <th>column</th>
                              {m.previewDiff.before.map((_, i) => <th key={`b${i}`}>before {m.previewDiff.before.length > 1 ? i + 1 : ''}</th>)}
                              {m.previewDiff.before.length === 0 && m.previewDiff.after.length > 0 && <th>before (absent)</th>}
                              {m.previewDiff.after.map((_, i) => <th key={`a${i}`}>after {m.previewDiff.after.length > 1 ? i + 1 : ''}</th>)}
                            </tr>
                          </thead>
                          <tbody>
                            {m.previewDiff.columns.map((col, ci) => (
                              <tr key={col + ci}>
                                <td>{col}</td>
                                {m.previewDiff.before.map((row, ri) => <td key={`b${ri}`}>{String(row[ci])}</td>)}
                                {m.previewDiff.before.length === 0 && m.previewDiff.after.length > 0 && <td>—</td>}
                                {m.previewDiff.after.map((row, ri) => <td key={`a${ri}`}>{String(row[ci])}</td>)}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        {m.previewDiff.rows_omitted && <p className="preview-note">Only the first {Math.max(m.previewDiff.before.length, m.previewDiff.after.length) || 0} changed rows are shown.</p>}
                      </div>
                    )}
                    {m.writeDone ? (
                      <p className="write-done">{m.writeReply || 'Write applied.'}</p>
                    ) : m.writeCancelled ? (
                      <p>{m.writeReply || 'Cancelled. No data was changed.'}</p>
                    ) : m.writeError ? (
                      <p className="write-error">{m.writeError}</p>
                    ) : (
                      <div className="write-actions">
                        <button className="write-confirm-btn" disabled={busy}
                          onClick={() => onConfirmWrite?.(m.id, m.writeToken)}>
                          {busy ? 'Applying…' : 'Confirm and apply change'}
                        </button>
                        <button className="write-cancel-btn" disabled={busy}
                          onClick={() => onCancelWrite?.(m.id, m.writeToken)}>
                          Cancel
                        </button>
                      </div>
                    )}
                    {m.followupSql && <SqlCard sql={m.followupSql} summary="Follow-up query" />}
                  </div>
                )}
                <ResultsTable results={m.results} />
                {m.kind === 'clarify' && (
                  <ClarifyCard
                    message={m}
                    onResolve={m.id === pendingId ? onResolve : null}
                  />
                )}
              </div>
              {m.detector && (m.kind === 'answer' || m.kind === 'clarify') && (
                <div className="detector-note" title={m.detector.reason || ''}>
                  <span className={`detector-tag ${m.clarification ? 'resolved' : String(m.detector.decision).toLowerCase()}`}>
                    {m.clarification ? 'RESOLVED' : m.detector.decision}
                  </span>
                  <span className="detector-label">
                    {m.detector.source === 'semantic_ir'
                      ? 'Semantic IR'
                      : m.detector.source === 'schema_meta'
                        ? 'Schema catalog'
                        : m.detector.source === 'rule_baseline'
                          ? 'Rule fallback'
                          : 'Hybrid ambiguity check'}
                    {m.clarification ? ` · ${m.detector.decision} → confirmed` : ''}
                  </span>
                  {m.detector.is_fallback && <span className="detector-rule">fallback</span>}
                  {Number.isFinite(Number(m.latency)) && (
                    <span className="detector-latency">{Number(m.latency).toFixed(2)}s</span>
                  )}
                </div>
              )}
            </div>
          </article>
        ))}
        {busy && (
          <article className="msg assistant">
            <div className="bubble assistant typing">
              <span className="dot" /><span className="dot" /><span className="dot" />
            </div>
          </article>
        )}
      </div>
    </div>
  )
})

export default Chat

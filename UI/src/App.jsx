import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Sidebar from './components/Sidebar.jsx'
import Chat from './components/Chat.jsx'
import Composer from './components/Composer.jsx'
import GradientBackdrop from './components/GradientBackdrop.jsx'
import { SUGGESTIONS, icons } from './components/icons.jsx'

const ACCEPT = '.csv,.tsv,.txt,.json,.db,.sqlite,.sqlite3'

let uid = 1
const nextId = () => `m${uid++}`

export default function App() {
  const [messages, setMessages] = useState([])
  const [busy, setBusy] = useState(false)
  const [schema, setSchema] = useState([])
  const [datasets, setDatasets] = useState([])
  const [uploadsDir, setUploadsDir] = useState('UI/uploads')
  const [pending, setPending] = useState(null) // { messageId, ambiguities, original, answers }
  const [pendingWrite, setPendingWrite] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [note, setNote] = useState(null)
  const [health, setHealth] = useState(null)

  const scrollRef = useRef(null)
  const fileRef = useRef(null)
  const dragDepth = useRef(0)
  // Stable per-tab conversation id for the worker's session memory (P1-4).
  const sessionIdRef = useRef(
    `s${Math.random().toString(36).slice(2, 10)}${Date.now().toString(36)}`)

  const refreshDatasets = useCallback(async () => {
    try {
      const res = await fetch('/api/datasets')
      const data = await res.json()
      setDatasets(data.datasets || [])
      setSchema(data.schema || [])
      if (data.uploadsDir) setUploadsDir(data.uploadsDir)
    } catch {
      /* dev server not ready */
    }
  }, [])

  useEffect(() => { refreshDatasets() }, [refreshDatasets])

  // The worker loads Qwen in the background; poll until it is ready or failed.
  useEffect(() => {
    let alive = true
    let timer = null
    const tick = async () => {
      try {
        const res = await fetch('/api/health')
        const data = await res.json()
        if (!alive) return
        setHealth(data)
        if (data.model?.state !== 'ready' && data.model?.state !== 'error') {
          timer = setTimeout(tick, 2000)
        }
      } catch {
        if (alive) timer = setTimeout(tick, 3000)
      }
    }
    tick()
    return () => { alive = false; clearTimeout(timer) }
  }, [])

  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
  }, [messages, busy])

  const push = (msg) => setMessages((m) => [...m, msg])

  const ask = useCallback(async (text, answers = null) => {
    const clean = String(text || '').trim()
    if (!clean) return
    setBusy(true)
    if (!answers) push({ id: nextId(), role: 'user', kind: 'text', text: clean })
    const progressId = nextId()
    push({ id: progressId, role: 'assistant', kind: 'progress',
      text: '', progressLabel: 'Starting model worker', tools: [] })

    const applyProgress = (progress) => {
      setMessages((current) => current.map((msg) => {
        if (msg.id !== progressId) return msg
        const tools = progress.type === 'tool' && progress.tool
          ? [...(msg.tools || []), progress.tool]
          : msg.tools
        return { ...msg, progressLabel: progress.label || msg.progressLabel, tools }
      }))
    }

    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/x-ndjson',
        },
        body: JSON.stringify({ message: clean, answers, sessionId: sessionIdRef.current }),
      })
      let data
      if (!res.ok) {
        const failure = await res.json()
        throw new Error(failure.error || `Request failed (${res.status})`)
      }
      if (res.headers.get('content-type')?.includes('application/x-ndjson') && res.body) {
        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''
        const consume = (line) => {
          if (!line.trim()) return
          const event = JSON.parse(line)
          if (event.type === 'progress') applyProgress(event.progress || {})
          else if (event.type === 'complete') data = event.data
          else if (event.type === 'error') throw new Error(event.error || 'Model request failed')
        }
        while (true) {
          const { value, done } = await reader.read()
          buffer += decoder.decode(value || new Uint8Array(), { stream: !done })
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''
          lines.forEach(consume)
          if (done) break
        }
        if (buffer.trim()) consume(buffer)
        if (!data) throw new Error('The model worker ended without a final response.')
      } else {
        data = await res.json()
      }
      const botMsg = {
        id: progressId,
        role: 'assistant',
        kind: data.status === 'clarify' ? 'clarify'
          : data.status === 'confirm_write' ? 'write_confirmation' : 'answer',
        text: data.reply,
        summary: data.summary,
        sql: data.sql,
        explanation: data.explanation,
        results: data.results,
        detector: data.detector,
        clarification: data.clarification,
        latency: data.latency_s,
        tools: data.tools || [],
        ambiguities: data.ambiguities || [],
        affectedRows: data.affected_rows,
        writeToken: data.write_token,
        previewDiff: data.preview || null,
        followupSql: data.followup_sql,
      }
      if (data.status === 'multi' && Array.isArray(data.answers)) {
        // one bubble per sub-answer; the progress bubble becomes the first
        const parts = data.answers.map((a, i) => ({
          id: i === 0 ? progressId : nextId(),
          role: 'assistant',
          kind: a.status === 'ok' && !a.error ? 'answer' : 'text',
          text: a.reply,
          sql: a.sql,
          explanation: a.explanation,
          results: a.results,
          detector: a.detector,
          latency: a.latency_s,
          tools: a.tools || [],
        }))
        setMessages((current) => [
          ...current.map((msg) => (msg.id === progressId ? parts[0] : msg)),
          ...parts.slice(1),
        ])
        setPending(null)
        return
      }
      setMessages((current) => current.map((msg) => msg.id === progressId ? botMsg : msg))
      if (data.status === 'clarify') {
        setPending({
          messageId: botMsg.id,
          ambiguities: data.ambiguities,
          original: clean,
          answers,
          decision: data.detector?.decision || null,
        })
      } else {
        setPending(null)
      }
      if (data.status === 'confirm_write') setPendingWrite(true)
      else if (data.status !== 'write_pending') setPendingWrite(false)
    } catch (e) {
      setMessages((current) => current.map((msg) => msg.id === progressId
        ? { id: progressId, role: 'assistant', kind: 'text', text: `Couldn't complete that: ${e.message}` }
        : msg))
      setPending(null)
    } finally {
      setBusy(false)
    }
  }, [])

  const confirmWriteAction = useCallback(async (messageId, token) => {
    if (!token || busy) return
    setBusy(true)
    try {
      const res = await fetch('/api/confirm-write', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || 'Could not apply the write.')
      setMessages((current) => current.map((msg) => msg.id === messageId
        ? { ...msg, writeDone: true, writeReply: data.reply, writeToken: null,
            followupSql: data.followup_sql || null, results: data.results || null }
        : msg))
      setPendingWrite(false)
      await refreshDatasets()
    } catch (e) {
      setMessages((current) => current.map((msg) => msg.id === messageId
        ? { ...msg, writeError: e.message, writeToken: null }
        : msg))
      setPendingWrite(false)
    } finally {
      setBusy(false)
    }
  }, [busy, refreshDatasets])

  const cancelWriteAction = useCallback(async (messageId, token) => {
    if (!token || busy) return
    setBusy(true)
    try {
      const res = await fetch('/api/cancel-write', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token, sessionId: sessionIdRef.current }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || 'Could not cancel the write preview.')
      setMessages((current) => current.map((msg) => msg.id === messageId
        ? { ...msg, writeCancelled: true, writeReply: data.reply, writeToken: null }
        : msg))
    } catch (e) {
      setMessages((current) => current.map((msg) => msg.id === messageId
        ? { ...msg, writeError: e.message, writeToken: null }
        : msg))
    } finally {
      setPendingWrite(false)
      setBusy(false)
    }
  }, [busy])

  const resolve = useCallback((questionId, optionIndex, optionLabel) => {
    if (!pending) return
    if (pending.decision === 'underspecified') {
      // Rephrase-style card: its chips are runnable example questions, not
      // interpretations — fire the clicked one as a fresh question.
      setPending(null)
      ask(optionLabel)
      return
    }
    const answers = { ...(pending.answers || {}), [questionId]: optionIndex }
    push({ id: nextId(), role: 'user', kind: 'choice', text: optionLabel })
    setMessages((m) =>
      m.map((msg) => (msg.id === pending.messageId ? { ...msg, resolved: optionLabel } : msg)),
    )
    setPending({ ...pending, answers })
    ask(pending.original, answers)
  }, [pending, ask])

  // Typed text while a clarify card is open: an AMBIGUOUS reading counts as a
  // custom interpretation (threaded into the SQL prompt server-side); an
  // UNDERSPECIFIED/rephrase card counts as a brand-new question. This is what
  // breaks the "clarify loop" the live UI could fall into.
  const send = useCallback((text) => {
    const t = String(text || '').trim()
    if (!t) return
    if (pending?.decision === 'underspecified') {
      setPending(null)
      ask(t)
      return
    }
    if (pending && pending.ambiguities?.length) {
      const answers = { ...(pending.answers || {}), [pending.ambiguities[0].id]: t }
      push({ id: nextId(), role: 'user', kind: 'choice', text: t })
      setMessages((m) => m.map((msg) => (
        msg.id === pending.messageId ? { ...msg, resolved: t } : msg
      )))
      setPending({ ...pending, answers })
      ask(pending.original, answers)
      return
    }
    ask(t)
  }, [pending, ask])

  // ---------- uploads ----------
  const uploadFiles = useCallback(async (files) => {
    if (!files?.length) return
    setUploading(true)
    setNote(null)
    try {
      for (const file of files) {
        const fd = new FormData()
        fd.append('file', file)
        const res = await fetch('/api/upload', { method: 'POST', body: fd })
        const data = await res.json()
        if (!res.ok) throw new Error(data.error || 'upload failed')
        setNote({ ok: true, text: `✓ ${data.message}` })
      }
      await refreshDatasets()
    } catch (e) {
      setNote({ ok: false, text: `✗ ${e.message}` })
    } finally {
      setUploading(false)
      if (fileRef.current) fileRef.current.value = ''
    }
  }, [refreshDatasets])

  // window-level drag & drop
  useEffect(() => {
    const over = (e) => {
      if (!e.dataTransfer?.types?.includes('Files')) return
      e.preventDefault()
      dragDepth.current += 1
      setDragging(true)
    }
    const leave = (e) => {
      e.preventDefault()
      dragDepth.current = Math.max(0, dragDepth.current - 1)
      if (dragDepth.current === 0) setDragging(false)
    }
    const drop = (e) => {
      e.preventDefault()
      dragDepth.current = 0
      setDragging(false)
      uploadFiles(e.dataTransfer?.files)
    }
    window.addEventListener('dragenter', over)
    window.addEventListener('dragover', over)
    window.addEventListener('dragleave', leave)
    window.addEventListener('drop', drop)
    return () => {
      window.removeEventListener('dragenter', over)
      window.removeEventListener('dragover', over)
      window.removeEventListener('dragleave', leave)
      window.removeEventListener('drop', drop)
    }
  }, [uploadFiles])

  const pickFile = () => fileRef.current?.click()
  const empty = messages.length === 0

  const tableCount = schema.length
  const rowCount = useMemo(() => schema.reduce((a, t) => a + (t.rows || 0), 0), [schema])

  return (
    <div className="app">
      <GradientBackdrop />

      <Sidebar
        datasets={datasets}
        schema={schema}
        uploadsDir={uploadsDir}
        note={note}
        uploading={uploading}
        health={health}
        onPick={pickFile}
        onRemove={async (stored) => {
          await fetch(`/api/datasets/${encodeURIComponent(stored)}`, { method: 'DELETE' })
          await refreshDatasets()
        }}
        onReset={async () => {
          await fetch('/api/reset', { method: 'POST' })
          setNote(null)
          await refreshDatasets()
        }}
      />

      <main className={`stage ${empty ? 'idle' : ''}`}>
        {empty ? (
          <section className="hero">
            <h1>Where should we start?</h1>
            <p className="hero-sub">
              Ambiguity-aware natural language → SQL with local Qwen. Ask anything;
              I'll clarify before I query.
            </p>

            <Composer
              disabled={busy || pendingWrite}
              pending={null}
              onSend={send}
              onAttach={pickFile}
              autoFocus
            />

            <ul className="suggestions">
              {SUGGESTIONS.map((s) => (
                <li key={s.text}>
                  <button className="suggestion" onClick={() => ask(s.text)}>
                    {icons[s.icon]}
                    <span>{s.text}</span>
                  </button>
                </li>
              ))}
            </ul>

            <p className="hero-hint">
              Drop a CSV anywhere (imported into the SQLite workspace) ·{' '}
              <button onClick={pickFile}>browse files</button>
            </p>
          </section>
        ) : (
          <>
            <header className="topbar">
              <div className="topbar-title">
                <h1>Clarion</h1>
                <span className="tagline">Ambiguity-Aware Natural Language → SQL</span>
              </div>
              <div className="topbar-stats">
                <span className="pill">{tableCount} table{tableCount === 1 ? '' : 's'}</span>
                <span className="pill">{rowCount.toLocaleString()} rows</span>
              </div>
            </header>

            <Chat
              ref={scrollRef}
              messages={messages}
              busy={busy}
              pendingId={pending?.messageId}
              onResolve={resolve}
              onConfirmWrite={confirmWriteAction}
              onCancelWrite={cancelWriteAction}
            />

            <Composer
              disabled={busy || pendingWrite}
              pending={pending}
              onSend={send}
              onResolve={resolve}
              onAttach={pickFile}
            />
          </>
        )}
      </main>

      <input
        ref={fileRef}
        type="file"
        accept={ACCEPT}
        multiple
        hidden
        onChange={(e) => uploadFiles(e.target.files)}
      />

      {dragging && (
        <div className="drop-overlay">
          <div className="drop-overlay-box">
            Drop to upload
            <small>CSV · TSV · JSON — imported as SQLite tables</small>
          </div>
        </div>
      )}
    </div>
  )
}

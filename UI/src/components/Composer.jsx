import { useEffect, useRef, useState } from 'react'
import { AttachIcon } from './icons.jsx'

export default function Composer({ disabled, pending, onSend, onAttach, autoFocus }) {
  const [text, setText] = useState('')
  const taRef = useRef(null)

  useEffect(() => {
    const ta = taRef.current
    if (!ta) return
    ta.style.height = 'auto'
    ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`
  }, [text])

  useEffect(() => {
    if (autoFocus) taRef.current?.focus()
  }, [autoFocus])

  function submit() {
    if (disabled || !text.trim()) return
    onSend(text)
    setText('')
  }

  return (
    <div className="composer">
      {pending && (
        <div className="pending-banner">
          <span className="pending-pulse" />
          Waiting for your clarification — pick an option above or type your own reading.
        </div>
      )}
      <div className="composer-box">
        {onAttach && (
          <button className="attach-btn" onClick={onAttach} title="Upload a dataset" aria-label="Upload a dataset">
            <AttachIcon />
          </button>
        )}
        <textarea
          ref={taRef}
          rows={1}
          value={text}
          placeholder={disabled ? 'Thinking…' : 'Ask anything about your data…'}
          disabled={disabled}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              submit()
            }
          }}
        />
        <button
          className="send-btn"
          onClick={submit}
          disabled={disabled || !text.trim()}
          aria-label="Send"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 19V5" /><path d="m5 12 7-7 7 7" />
          </svg>
        </button>
      </div>
      <div className="composer-hint">
        Enter to send · Shift+Enter for a newline · ambiguity is clarified before SQL is written
      </div>
    </div>
  )
}

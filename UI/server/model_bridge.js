// Long-lived bridge to `model_worker.py`.
//
// Spawning Python per request would reload Qwen every time; instead the Vite
// server keeps one worker alive and talks JSON-lines over stdin/stdout. The
// first `warmup` request loads the weights while the page is still opening.
import { spawn } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { PROJECT_ROOT, findPython } from './python.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const WORKER = path.join(__dirname, 'model_worker.py')
const WARMUP_TIMEOUT_MS = 5 * 60 * 1000
const QUERY_TIMEOUT_MS = 5 * 60 * 1000

export class ModelBridge {
  constructor() {
    this.child = null
    this.nextId = 1
    this.pending = new Map()
    this.status = 'idle'
    this.error = null
    this.info = null
    this.stderrTail = []
    this.warmupPromise = null
    this.queue = Promise.resolve()
  }

  start() {
    if (this.warmupPromise) return this.warmupPromise
    this._spawn()
    this.status = 'loading'
    this.warmupPromise = this._send({ action: 'warmup' }, WARMUP_TIMEOUT_MS)
      .then((data) => {
        this.status = 'ready'
        this.error = null
        this.info = data
        return data
      })
      .catch((err) => {
        this.status = 'error'
        this.error = err.message
        throw err
      })
    // A failed warmup must not become an unhandled rejection: /api/health and
    // the next /api/query both report it explicitly.
    this.warmupPromise.catch(() => {})
    return this.warmupPromise
  }

  /** Operations that call the model are serialized (one GPU, one model). */
  query(payload, onProgress = null) {
    this.start()
    const run = () => this._send(
      { action: 'query', ...payload, stream: Boolean(onProgress) },
      QUERY_TIMEOUT_MS,
      onProgress,
    )
    const request = this.queue.then(run, run)
    this.queue = request.then(() => undefined, () => undefined)
    return request
  }

  confirmWrite(token) {
    this.start()
    const run = () => this._send(
      { action: 'confirm_write', token }, QUERY_TIMEOUT_MS,
    )
    const request = this.queue.then(run, run)
    this.queue = request.then(() => undefined, () => undefined)
    return request
  }

  cancelWrite(token) {
    this.start()
    const run = () => this._send(
      { action: 'cancel_write', token }, QUERY_TIMEOUT_MS,
    )
    const request = this.queue.then(run, run)
    this.queue = request.then(() => undefined, () => undefined)
    return request
  }

  describe() {
    return {
      state: this.status,
      error: this.error,
      info: this.info,
      pid: this.child?.pid ?? null,
    }
  }

  stop() {
    const child = this.child
    this.child = null
    this.warmupPromise = null
    this.status = 'idle'
    if (!child) return
    try { child.stdin.end() } catch { /* already closed */ }
    try { child.kill('SIGTERM') } catch { /* already gone */ }
    this._rejectAll(new Error('model worker stopped'))
  }

  _spawn() {
    if (this.child) return this.child

    const python = findPython()
    const child = spawn(python, [WORKER], {
      cwd: PROJECT_ROOT,
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1',
        TOKENIZERS_PARALLELISM: 'false',
      },
      stdio: ['pipe', 'pipe', 'pipe'],
    })
    this.child = child
    this.stderrTail = []

    let buffer = ''
    child.stdout.setEncoding('utf8')
    child.stdout.on('data', (chunk) => {
      buffer += chunk
      let newline
      while ((newline = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, newline).trim()
        buffer = buffer.slice(newline + 1)
        if (line) this._onLine(line)
      }
    })

    child.stderr.setEncoding('utf8')
    child.stderr.on('data', (chunk) => {
      const text = String(chunk)
      this.stderrTail.push(text)
      while (this.stderrTail.length > 60) this.stderrTail.shift()
      process.stderr.write(text)
    })

    child.on('error', (err) => {
      this.status = 'error'
      this.error = `Could not start ${python}: ${err.message}`
      this.child = null
      this._rejectAll(new Error(this.error))
    })

    child.on('exit', (code, signal) => {
      if (this.child !== child) return
      this.child = null
      this.warmupPromise = null
      if (this.status !== 'idle') {
        this.status = 'error'
        const tail = this.stderrTail.join('').trim().split('\n').slice(-4).join('\n')
        this.error = `Model worker exited (${signal || `code ${code}`}).`
          + (tail ? `\n${tail}` : '')
      }
      this._rejectAll(new Error(this.error || 'model worker exited'))
    })

    return child
  }

  _onLine(line) {
    let message
    try {
      message = JSON.parse(line)
    } catch {
      process.stderr.write(`[model] non-JSON output: ${line}\n`)
      return
    }
    const waiter = this.pending.get(message.id)
    if (!waiter) return
    if (message.progress) {
      waiter.onProgress?.(message.progress)
      return
    }
    this.pending.delete(message.id)
    clearTimeout(waiter.timer)
    if (message.ok) waiter.resolve(message.data)
    else {
      const err = new Error(message.error || 'model request failed')
      err.code = message.code
      waiter.reject(err)
    }
  }

  _send(payload, timeoutMs, onProgress = null) {
    const child = this._spawn()
    return new Promise((resolve, reject) => {
      const id = this.nextId++
      const timer = setTimeout(() => {
        this.pending.delete(id)
        reject(new Error(`Model request timed out after ${Math.round(timeoutMs / 1000)}s.`))
      }, timeoutMs)

      this.pending.set(id, { resolve, reject, timer, onProgress })
      try {
        child.stdin.write(JSON.stringify({ id, ...payload }) + '\n')
      } catch (err) {
        this.pending.delete(id)
        clearTimeout(timer)
        reject(err)
      }
    })
  }

  _rejectAll(error) {
    for (const waiter of this.pending.values()) {
      clearTimeout(waiter.timer)
      waiter.reject(error)
    }
    this.pending.clear()
  }
}

export const modelBridge = new ModelBridge()

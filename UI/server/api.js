// Vite middleware plugin: uploads → SQLite workspace → the real Qwen pipeline.
//
//   POST   /api/upload        multipart upload, imported as a SQLite table
//   GET    /api/datasets      registry + live schema read from the workspace DB
//   DELETE /api/datasets/:id  remove a file and its table
//   GET    /api/health        model-worker state (loading / ready / error)
//   POST   /api/query         ambiguity detection → clarification → SQL → rows
//   POST   /api/reset         clear uploads and restore the demo database
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import busboy from 'busboy'
import { modelBridge } from './model_bridge.js'
import { PROJECT_ROOT } from './python.js'
import {
  UPLOAD_DIR,
  WORKSPACE_DB,
  dropTable,
  ensureWorkspace,
  importTable,
  importSQLiteFile,
  listSchema,
  parseUploadFile,
  readWorkspaceInfo,
  resetWorkspace,
  safeTable,
  uniqueTable,
  tablesForDataset,
} from './workspace.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const REGISTRY = path.join(UPLOAD_DIR, 'registry.json')

const MAX_BYTES = 25 * 1024 * 1024
const ALLOWED = new Set(['.csv', '.tsv', '.txt', '.json', '.db', '.sqlite', '.sqlite3'])

// ---------- registry helpers ----------
function ensureDirs() {
  fs.mkdirSync(UPLOAD_DIR, { recursive: true })
  if (!fs.existsSync(REGISTRY)) fs.writeFileSync(REGISTRY, '[]')
  ensureWorkspace()
}

function readRegistry() {
  ensureDirs()
  try {
    const list = JSON.parse(fs.readFileSync(REGISTRY, 'utf8'))
    return Array.isArray(list) ? list : []
  } catch {
    return []
  }
}

function writeRegistry(list) {
  ensureDirs()
  fs.writeFileSync(REGISTRY, JSON.stringify(list, null, 2))
}

function publicSchema() {
  const uploads = new Set(readRegistry().flatMap(tablesForDataset))
  return listSchema(uploads)
}

function json(res, code, obj) {
  const body = JSON.stringify(obj)
  res.writeHead(code, { 'Content-Type': 'application/json; charset=utf-8' })
  res.end(body)
}

function readBody(req, limit = 1024 * 1024) {
  return new Promise((resolve, reject) => {
    let size = 0
    const chunks = []
    req.on('data', (chunk) => {
      size += chunk.length
      if (size > limit) { reject(new Error('Payload too large')); req.destroy(); return }
      chunks.push(chunk)
    })
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')))
    req.on('error', reject)
  })
}

function uniqueName(filename) {
  const ext = path.extname(filename).toLowerCase()
  const stem = path.basename(filename, path.extname(filename)).replace(/[^a-zA-Z0-9_-]+/g, '_') || 'dataset'
  let candidate = `${stem}${ext}`
  let i = 1
  while (fs.existsSync(path.join(UPLOAD_DIR, candidate))) candidate = `${stem}_${i++}${ext}`
  return candidate
}

// ---------- the plugin ----------
function middleware() {
  return async (req, res, next) => {
    try {
      const url = new URL(req.url, 'http://localhost')
      const p = url.pathname

      if (p === '/api/upload' && req.method === 'POST') return handleUpload(req, res)

      if (p === '/api/datasets' && req.method === 'GET') {
        const datasets = readRegistry()
        return json(res, 200, {
          uploadsDir: path.relative(PROJECT_ROOT, UPLOAD_DIR) || UPLOAD_DIR,
          datasets,
          schema: publicSchema(),
        })
      }

      if (p.startsWith('/api/datasets/') && req.method === 'DELETE') {
        const stored = decodeURIComponent(p.slice('/api/datasets/'.length))
        const list = readRegistry()
        const idx = list.findIndex((entry) => entry.stored === stored)
        if (idx === -1) return json(res, 404, { error: 'Not found' })
        const [entry] = list.splice(idx, 1)
        for (const table of tablesForDataset(entry)) {
          try { dropTable(table) } catch { /* table may already be gone */ }
        }
        writeRegistry(list)
        try { fs.unlinkSync(path.join(UPLOAD_DIR, stored)) } catch { /* already gone */ }
        return json(res, 200, { ok: true, removed: entry.original })
      }

      if (p === '/api/health' && req.method === 'GET') {
        return json(res, 200, {
          model: modelBridge.describe(),
          workspace: readWorkspaceInfo(),
        })
      }

      if (p === '/api/confirm-write' && req.method === 'POST') {
        const body = JSON.parse((await readBody(req)) || '{}')
        if (typeof body.token !== 'string' || !body.token) {
          return json(res, 400, { error: 'Missing write confirmation token.' })
        }
        try {
          const outcome = await modelBridge.confirmWrite(body.token)
          return json(res, 200, outcome)
        } catch (err) {
          return json(res, err.code === 'input' ? 400 : 500,
            { error: err.message, code: err.code || 'write_error' })
        }
      }

      if (p === '/api/cancel-write' && req.method === 'POST') {
        const body = JSON.parse((await readBody(req)) || '{}')
        if (typeof body.token !== 'string' || !body.token) {
          return json(res, 400, { error: 'Missing write confirmation token.' })
        }
        try {
          const outcome = await modelBridge.cancelWrite(body.token)
          return json(res, 200, outcome)
        } catch (err) {
          return json(res, err.code === 'input' ? 400 : 500,
            { error: err.message, code: err.code || 'write_error' })
        }
      }

      if (p === '/api/query' && req.method === 'POST') {
        const body = JSON.parse((await readBody(req)) || '{}')
        const message = typeof body.message === 'string' ? body.message : ''
        if (!message.trim()) {
          return json(res, 400, { error: 'Please enter a question.' })
        }

        const stream = req.headers.accept?.includes('application/x-ndjson')
        if (stream) {
          res.writeHead(200, {
            'Content-Type': 'application/x-ndjson; charset=utf-8',
            'Cache-Control': 'no-cache, no-transform',
            'X-Accel-Buffering': 'no',
          })
          res.write(JSON.stringify({ type: 'progress', progress: {
            type: 'stage', label: 'Starting model worker',
          } }) + '\n')
        }

        try {
          const outcome = await modelBridge.query({
            message,
            answers: body.answers && typeof body.answers === 'object' ? body.answers : {},
            dbPath: WORKSPACE_DB,
          }, stream ? (progress) => {
            if (!res.destroyed) res.write(JSON.stringify({ type: 'progress', progress }) + '\n')
          } : null)
          if (stream) {
            res.end(JSON.stringify({ type: 'complete', data: outcome }) + '\n')
            return
          }
          return json(res, 200, outcome)
        } catch (err) {
          const code = err.code || 'model_error'
          const status = code === 'input' ? 400 : code === 'model_missing' ? 503 : 500
          if (stream) {
            res.end(JSON.stringify({ type: 'error', error: err.message, code }) + '\n')
            return
          }
          return json(res, status, { error: err.message, code })
        }
      }

      if (p === '/api/reset' && req.method === 'POST') {
        for (const entry of readRegistry()) {
          try { fs.unlinkSync(path.join(UPLOAD_DIR, entry.stored)) } catch { /* ignore */ }
        }
        writeRegistry([])
        resetWorkspace()
        return json(res, 200, { ok: true, schema: publicSchema() })
      }

      return next()
    } catch (err) {
      return json(res, 500, { error: String(err?.message || err) })
    }
  }
}

function install(server) {
  ensureDirs()
  modelBridge.start()
  server.httpServer?.on('close', () => modelBridge.stop())
  server.middlewares.use(middleware())
}

export function clarionApi() {
  return {
    name: 'clarion-api',
    configureServer(server) { install(server) },
    configurePreviewServer(server) { install(server) },
  }
}

function handleUpload(req, res) {
  ensureDirs()
  let bb
  try {
    bb = busboy({ headers: req.headers, limits: { fileSize: MAX_BYTES, files: 1 } })
  } catch (e) {
    return json(res, 400, { error: `Bad upload: ${e.message}` })
  }

  const saved = []
  let failed = null
  const writes = []

  bb.on('file', (name, stream, info) => {
    const original = info.filename || 'dataset.csv'
    const ext = path.extname(original).toLowerCase()
    if (!ALLOWED.has(ext)) {
      failed = `Unsupported file type "${ext || '(none)'}" — allowed: ${[...ALLOWED].join(', ')}`
      stream.resume()
      return
    }
    const stored = uniqueName(original)
    const dest = path.join(UPLOAD_DIR, stored)
    writes.push(
      new Promise((resolve) => {
        const ws = fs.createWriteStream(dest)
        let bytes = 0
        stream.on('data', (chunk) => { bytes += chunk.length })
        stream.on('limit', () => {
          failed = 'File exceeds the 25 MB limit.'
          ws.destroy()
          fs.rmSync(dest, { force: true })
        })
        stream.pipe(ws)
        ws.on('close', () => {
          if (!failed) saved.push({ original, stored, size: bytes, ext })
          resolve()
        })
        ws.on('error', (err) => {
          failed = `Could not write file: ${err.message}`
          resolve()
        })
      }),
    )
  })

  bb.on('error', (err) => { failed = err.message })

  bb.on('close', async () => {
    await Promise.all(writes)
    if (failed) return json(res, 400, { error: failed })
    if (!saved.length) return json(res, 400, { error: 'No file received.' })

    const file = saved[0]
    const dest = path.join(UPLOAD_DIR, file.stored)
    const list = readRegistry()
    // Re-uploading a file with the same name replaces its previous import:
    // otherwise a stale duplicate table keeps old rows queryable alongside
    // the new ones and the model can answer from either.
    const previous = list.findIndex((entry) => entry.original === file.original)
    const [oldEntry] = previous === -1 ? [] : list.splice(previous, 1)
    const dropTables = oldEntry ? tablesForDataset(oldEntry) : []
    let parsed
    let importedTables
    try {
      const base = safeTable(path.basename(file.original, file.ext))
      if (['.db', '.sqlite', '.sqlite3'].includes(file.ext)) {
        importedTables = importSQLiteFile(dest, base, dropTables)
      } else {
        parsed = parseUploadFile(dest, file.ext)
        if (!parsed.columns.length) throw new Error('no parsable columns (expected a header row)')
        // a same-name replacement reuses the previous table name (its old
        // copy is dropped inside the import transaction, so it is free again)
        const table = dropTables.length === 1 && dropTables[0] === base
          ? base : uniqueTable(base)
        importTable(table, parsed, dropTables)
        importedTables = [{ name: table, sourceName: table,
          rows: parsed.rows.length, columns: parsed.columns }]
      }
    } catch (e) {
      fs.rmSync(dest, { force: true })
      // restore the previous import's registry entry — its tables were only
      // dropped inside the failed import's transaction, so they still exist
      if (oldEntry) { list.push(oldEntry); writeRegistry(list) }
      return json(res, 400, { error: `Could not import that file: ${e.message}` })
    }
    if (oldEntry) {
      try { fs.unlinkSync(path.join(UPLOAD_DIR, oldEntry.stored)) } catch { /* already gone */ }
    }

    const entry = {
      ...file,
      table: importedTables[0].name,
      tables: importedTables,
      rows: importedTables.reduce((total, table) => total + table.rows, 0),
      columns: importedTables.flatMap((table) => table.columns),
      uploadedAt: new Date().toISOString(),
    }
    list.push(entry)
    writeRegistry(list)

    return json(res, 200, {
      ok: true,
      dataset: entry,
      message: `Imported ${entry.original} as ${importedTables.length} SQLite table${importedTables.length === 1 ? '' : 's'} — the model can query them now.`,
      schema: publicSchema(),
    })
  })

  req.pipe(bb)
}
